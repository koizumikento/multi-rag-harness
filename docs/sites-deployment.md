# Sites MCP adapter and private Python backend

Sites runs the dependency-free Worker in `sites/`; Python remains the owner of
all 16 existing tools, schemas, retrieval methods, reranking, persistence, graph
traversal and provenance. Codex remains the client. This is not a Sites-only
deployment: Workers' 128 MB isolate and HTTP-only external connections cannot
host the local model weights and SQLite/embedded Qdrant/Kuzu durable stores.
Postgres/pgvector remain the existing unimplemented adapters, not a new remote
storage solution. The Python library, CLI and default stdio transport remain
available.

## Access and isolation contract

- Deploy **one backend process and one data directory per Site/owner**. Use a
  dedicated remote Qdrant collection and credentials too if selecting remote
  Qdrant. Never point this deployment at an existing/shared corpus or customer
  database without separate authorization to expose it.
- Sites Dispatch owns OAuth/authentication and supplies
  `oai-authenticated-user-id`. Configure that Site's stable user ID in both
  Worker and backend. Email is not an authorization key. The raw Worker must
  only be served through Sites Dispatch; a direct public Worker deployment
  would let callers spoof identity headers.
- Worker rejects missing identity (401) and a different owner (403) on every
  request. `OAI-Sites-Authorization` service access alone grants no user data
  access. Worker replaces inbound service credentials and identity headers with
  its runtime token and authenticated owner ID.
- Backend independently validates the service token and owner on every
  discovery, read, graph traversal and write. All object IDs belong to that
  isolated tenant. Retrieval `scope`/filters do not provide authorization.
  Do not share this backend with another Site or identity. Shared workspaces
  require a separately designed membership/object authorization layer.
- `rag_ingest_path` keeps its existing schema, but HTTP restricts it to an
  existing absolute, dedicated corpus root (including checks on nested links).
  Data storage must be outside that root. Only trusted operators may write this
  directory; keep credentials, model cache, configuration and unrelated files
  outside it. Mount the corpus read-only to the backend account where possible.
  Filesystem confinement is not a sandbox against an operator changing links
  during ingestion.
- No client-selected upstream URL, shell forwarding, CORS, session IDs, or
  automatic write retries. A 1 MiB request limit applies at both boundaries.
  Existing result bounds and error shapes come from Python. The Worker uses
  stateless POST `/mcp`, preserves JSON-RPC/structured tool results, and returns
  405 for GET/DELETE (no SSE listening). It validates browser Origin and does not
  forward browser cookies, client bearer tokens, or backend debug headers.

## Configure and run backend

Install from this repository with `uv sync --frozen`. Provision dedicated empty
corpus/storage directories on an always-on machine with capacity for the
selected local models. Preload models deliberately if offline service is
required; the existing local adapters download weights on first use. Keep the
same pinned embedding dimension for the lifetime of the collection. Run a
single process (embedded Qdrant locks its directory).

Set these environment variable **names** via the backend's secret manager or
untracked environment, without printing their values:

```text
MRH_DATA_DIR
MRH_HTTP_INGEST_ROOT
MRH_HTTP_ALLOWED_HOST
MRH_HTTP_BACKEND_TOKEN
MRH_HTTP_SITE_USER_ID
```

`MRH_DATA_DIR` and `MRH_HTTP_INGEST_ROOT` must be separate absolute paths for this
tenant. `MRH_HTTP_ALLOWED_HOST` is the exact HTTPS ingress Host (including port
when non-default). Use a random backend token of at least 32 characters. This
token is service authority; the Site user ID supplies the separately checked
actor. Existing optional provider keys remain backend-only environment secrets.
Do not put any secret in TOML, source, logs, model prompts or the manifest.

```bash
uv run multi-rag-harness serve-http --port 8000
```

The backend binds `127.0.0.1`, disables access logging, and opens storage once
for the process lifetime. Put it behind a stable HTTPS reverse proxy or a
[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/)
on that machine. A named tunnel's published hostname maps to
`http://127.0.0.1:8000`; set `MRH_HTTP_ALLOWED_HOST` to the public hostname and
expose only `/mcp`, with an HTTP 404 catch-all. Keep the tunnel credential on
the backend machine. The existing application token protects this HTTP ingress;
do not remove it. Quick/ephemeral tunnels are not a production endpoint. No
account, paid contract or tunnel is created by this PR. A provisioned hostname,
TLS ingress, backend machine, backups, models and secret storage are operator
dependencies. Sites can reach the HTTPS endpoint with Worker `fetch`; private
LAN URLs/raw TCP are not reachable directly.

## Prepare Sites source (publication is a separate action)

The contract matches the installed official Sites 1.0.0 Worker ESM starter:
`dist/server/index.js` exports `default.fetch(request, env, ctx)` and
`dist/.openai/hosting.json` declares `capabilities: ["mcp"]`, with no `static`,
D1 or R2. The root of the Sites source checkout is **`sites/`**, not the Python
repository; only the Worker and manifest are packaged into its build output.
The artifact validator rejects extra files in that output.

```bash
cd sites
npm test
npm run build
npm run validate
```

No Node dependencies or installation are needed. A later authorized publication
uses the official Sites source helper/workflow on a clean copy of `sites/`.
Register a new private Site, persist its returned `project_id` with the official
`set-project-id.mjs` helper, and preserve that identity on every later update.
Do not use a placeholder Site ID or overwrite an existing Site for testing.
The checked-in manifest deliberately has no registered project ID.

Configure the following **runtime secret names** via native Sites tools:

```text
MRH_BACKEND_URL
MRH_BACKEND_TOKEN
MRH_SITE_USER_ID
```

The URL is the fixed HTTPS endpoint ending exactly in `/mcp`; no credentials,
query or fragment. The token and user ID must match backend configuration.
Obtain the user's Site-scoped ID from authenticated server-side identity
(an operator-only setup step); don't infer it from another Site/email or add a
public identity-debug tool. Keep local `.env` files ignored; `.env.example`
contains names only. The Worker never redirects a backend request and times
out after 120 seconds. After timeout/disconnection, check durable write outcome
before retrying: a failed response does not prove a write was rolled back.

For native workflow packaging, run these ordered argument arrays from the
Sites source root and retain the verified archive/commit:

```json
[
  ["node", "--test", "tests/worker.test.mjs"],
  ["node", "scripts/build.mjs"],
  ["node", "scripts/validate-artifact.mjs"]
]
```

Sites owns the generated App/plugin and OAuth; no local plugin registration or
new OAuth server is needed. Preserve private audience. Once authorized and
deployed, inspect `get_site(include_mcp_connection: true)` and confirm
`has_mcp`, connect its returned plugin, verify authenticated initialization,
`tools/list`, a source-grounded read and an isolated write. An HTTP 200/build
alone does not prove discovery or client acceptance. Confirm service-only and
other-user requests fail. This PR does not perform those live acceptance steps.

## Reproducible verification and cleanup

```bash
uv lock --check
uv run ruff check
uv run ruff format --check
uv run ty check
uv run coverage run -m pytest
uv run coverage report
uv build --sdist --wheel --out-dir work/dist
cd sites
npm test
npm run build
npm run validate
```

The Python HTTP fixtures use tmp-dir real SQLite/Qdrant/Kuzu plus fake models
and fake Codex. They compare all tool discovery schemas with stdio, exercise
every tool, compare read results, retain provenance, and check tenant denial,
invalid protocol/schema and confined ingestion. Worker mocks assert exact
protocol forwarding, header/authority separation and safe failures. They use
no real model/service credentials or customer data. These are local/isolated
proofs, distinct from CI and live Sites/client acceptance.

Keep reproducible tests/source. Stop only test servers/tunnels created for the
run and remove only their isolated database directories, build output,
archives and dependency environments after verification. Record owned paths
and process IDs before starting, verify resolved paths before recursive
deletion, and check residual state afterward. Preserve shared caches, model
weights, existing services/data, the PR branch and its review worktree.

Official references checked 2026-10-10:
[MCP Streamable HTTP 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports),
[Python SDK v1.x](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x).
This repo's `uv.lock` pins MCP 1.28.1; it uses the v1 FastMCP API, not the SDK
main branch's newer v2/2026 transport. Compatibility target is stateless
2025-11-25 JSON responses with initialization/discovery/calls. Other protocol
versions follow the pinned SDK's supported-version negotiation; live Sites
client compatibility remains to be verified after deployment.
