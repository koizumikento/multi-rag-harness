const MAX_REQUEST_BYTES = 1024 * 1024;

function error(status, message) {
  return Response.json({ error: message }, {
    status,
    headers: { "Cache-Control": "no-store", ...(status === 405 ? { Allow: "POST" } : {}) },
  });
}

export default {
  async fetch(request, env) {
    const site = new URL(request.url);
    if (site.pathname !== "/mcp") return error(404, "Not found");
    // Dispatch authenticates and replaces identity headers on hosted Sites.
    const user = request.headers.get("oai-authenticated-user-id");
    if (!user) return error(401, "ChatGPT sign-in required");
    if (!env.MRH_SITE_USER_ID || !env.MRH_BACKEND_TOKEN || !env.MRH_BACKEND_URL) {
      return error(503, "Sites backend is not configured");
    }
    if (user !== env.MRH_SITE_USER_ID) return error(403, "Site user is not authorized");
    const origin = request.headers.get("Origin");
    if (origin && origin !== site.origin) return error(403, "Invalid Origin");
    if (request.method !== "POST") return error(405, "Use POST /mcp");
    if (request.headers.get("Content-Type")?.split(";")[0].trim().toLowerCase() !== "application/json") {
      return error(415, "Use application/json");
    }
    let backend;
    try {
      backend = new URL(env.MRH_BACKEND_URL);
      if (backend.protocol !== "https:" || backend.username || backend.password ||
          backend.pathname !== "/mcp" || backend.search || backend.hash ||
          env.MRH_BACKEND_TOKEN.length < 32) {
        return error(503, "Invalid backend configuration");
      }
    } catch {
      return error(503, "Invalid backend configuration");
    }

    const chunks = [];
    let length = 0;
    const reader = request.body?.getReader();
    try {
      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          length += value.byteLength;
          if (length > MAX_REQUEST_BYTES) {
            await reader.cancel();
            return error(413, "Request too large");
          }
          chunks.push(value);
        }
      }
    } catch {
      return error(400, "Unable to read request body");
    }
    const body = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      body.set(chunk, offset);
      offset += chunk.byteLength;
    }
    // Never forward client auth, cookies, URL parameters, or identity overrides.
    const headers = new Headers({
      "Content-Type": "application/json",
      Accept: request.headers.get("Accept") || "application/json, text/event-stream",
      Authorization: `Bearer ${env.MRH_BACKEND_TOKEN}`,
      "X-MRH-Site-User-ID": user,
    });
    const version = request.headers.get("MCP-Protocol-Version");
    if (version) headers.set("MCP-Protocol-Version", version);
    for (const name of ["Mcp-Method", "Mcp-Name"]) {
      const value = request.headers.get(name);
      if (value !== null) headers.set(name, value);
    }
    try {
      const result = await fetch(backend, {
        method: "POST", headers, body, redirect: "manual",
        signal: AbortSignal.timeout(120_000),
      });
      if (result.status === 401 || result.status === 403) {
        return error(result.status, "Backend access denied");
      }
      if ((result.status >= 300 && result.status < 400) || result.status >= 500) {
        return error(502, "Backend unavailable; check configuration before retrying");
      }
      const outputHeaders = new Headers({ "Cache-Control": "no-store" });
      outputHeaders.set("Content-Type", result.headers.get("Content-Type") || "application/json");
      return new Response(result.body, { status: result.status, headers: outputHeaders });
    } catch {
      return error(502, "Backend unavailable; confirm write outcome before retrying");
    }
  },
};
