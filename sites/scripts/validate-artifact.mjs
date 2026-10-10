import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";

const root = new URL("../dist/", import.meta.url);
assert.deepEqual((await readdir(root)).sort(), [".openai", "server"]);
assert.deepEqual(await readdir(new URL("server/", root)), ["index.js"]);
assert.deepEqual(await readdir(new URL(".openai/", root)), ["hosting.json"]);
const manifest = JSON.parse(await readFile(new URL(".openai/hosting.json", root), "utf8"));
assert.deepEqual(manifest.capabilities, ["mcp"]);
assert.equal(manifest.static, undefined);
const source = await readFile(new URL("server/index.js", root), "utf8");
const worker = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
assert.equal(typeof worker.default.fetch, "function");
console.log("Sites Worker ESM artifact validated");
