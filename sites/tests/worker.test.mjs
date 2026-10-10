import assert from "node:assert/strict";
import { test, afterEach } from "node:test";
import worker from "../worker/index.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });
const env = {
  MRH_BACKEND_URL: "https://backend.example/mcp",
  MRH_BACKEND_TOKEN: "fixture-service-token-32-characters",
  MRH_SITE_USER_ID: "site-owner",
};
function request(body = { jsonrpc: "2.0", id: 1, method: "tools/list" }, headers = {}, method = "POST") {
  return new Request("https://site.example/mcp", {
    method,
    headers: { "Content-Type": "application/json", "oai-authenticated-user-id": "site-owner", ...headers },
    ...(method === "POST" ? { body: JSON.stringify(body) } : {}),
  });
}

test("missing identity, service-only access, and other users never reach backend", async () => {
  globalThis.fetch = () => { throw new Error("must not be called"); };
  for (const [headers, status] of [
    [{ "oai-authenticated-user-id": "", "OAI-Sites-Authorization": "Bearer bypass" }, 401],
    [{ "oai-authenticated-user-id": "other-user" }, 403],
    [{ Origin: "https://attacker.example" }, 403],
  ]) assert.equal((await worker.fetch(request(undefined, headers), env)).status, status);
});

test("forwards exact protocol body and bounded headers using runtime authority", async () => {
  const message = { jsonrpc: "2.0", id: "fixture", method: "tools/call", params: {
    name: "rag_search", arguments: { query: "source", filters: { scopes: ["team"] }, top_k: 5 },
  } };
  const result = { jsonrpc: "2.0", id: "fixture", result: {
    content: [{ type: "text", text: "source" }], structuredContent: { results: [{ source_id: "d1" }] },
  } };
  globalThis.fetch = async (url, init) => {
    assert.equal(url.href, env.MRH_BACKEND_URL);
    assert.equal(init.redirect, "manual");
    assert.deepEqual(JSON.parse(new TextDecoder().decode(init.body)), message);
    assert.equal(init.headers.get("Authorization"), `Bearer ${env.MRH_BACKEND_TOKEN}`);
    assert.equal(init.headers.get("X-MRH-Site-User-ID"), "site-owner");
    assert.equal(init.headers.get("MCP-Protocol-Version"), "2025-11-25");
    for (const header of ["Cookie", "Origin", "MCP-Session-ID", "OAI-Sites-Authorization"]) {
      assert.equal(init.headers.get(header), null);
    }
    return Response.json(result, { headers: { "Set-Cookie": "private", "X-Debug": "private" } });
  };
  const response = await worker.fetch(request(message, {
    Authorization: "Bearer client-token", Cookie: "private", "X-MRH-Site-User-ID": "attacker",
    "MCP-Protocol-Version": "2025-11-25", "MCP-Session-ID": "client-session",
  }), env);
  assert.deepEqual(await response.json(), result);
  assert.equal(response.headers.get("Set-Cookie"), null);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
});

test("notifications and SDK protocol/tool errors remain intact", async () => {
  for (const status of [202, 400, 406]) {
    globalThis.fetch = async () => new Response(status === 202 ? null : '{"error":"invalid"}', { status });
    assert.equal((await worker.fetch(request(), env)).status, status);
  }
});

test("redirects, backend errors, and transport exceptions disclose no private content", async () => {
  for (const status of [302, 401, 403, 500]) {
    globalThis.fetch = async () => new Response("private token and stack", {
      status, headers: { Location: "https://attacker.example" },
    });
    const result = await worker.fetch(request(), env);
    assert.equal(result.status, status === 401 || status === 403 ? status : 502);
    assert.ok(!(await result.text()).includes("private"));
  }
  globalThis.fetch = async () => { throw new Error("secret token"); };
  const result = await worker.fetch(request(), env);
  assert.equal(result.status, 502);
  assert.ok((await result.text()).includes("confirm write outcome"));
});

test("invalid config, content type, method, path and large bodies fail closed", async () => {
  globalThis.fetch = () => { throw new Error("must not be called"); };
  for (const url of ["http://backend.example/mcp", "https://backend.example/arbitrary", "https://u:p@backend.example/mcp", "invalid"]) {
    assert.equal((await worker.fetch(request(), { ...env, MRH_BACKEND_URL: url })).status, 503);
  }
  assert.equal((await worker.fetch(request(), {})).status, 503);
  assert.equal((await worker.fetch(request(undefined, { "Content-Type": "text/plain" }), env)).status, 415);
  assert.equal((await worker.fetch(request(undefined, {}, "GET"), env)).status, 405);
  assert.equal((await worker.fetch(new Request("https://site.example/other"), env)).status, 404);
  assert.equal((await worker.fetch(request("a".repeat(1024 * 1024)), env)).status, 413);
});

test("aborted request stream returns a bounded error", async () => {
  const body = new ReadableStream({ start(controller) { controller.error(new Error("private detail")); } });
  const input = new Request("https://site.example/mcp", {
    method: "POST", body, duplex: "half",
    headers: { "Content-Type": "application/json", "oai-authenticated-user-id": "site-owner" },
  });
  const result = await worker.fetch(input, env);
  assert.equal(result.status, 400);
  assert.ok(!(await result.text()).includes("private"));
});
