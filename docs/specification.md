# multi-rag-harness Specification

## Purpose

`multi-rag-harness` is a local-first retrieval and memory substrate for Codex
agents. It exposes retrieval, memory, graph, reranking, and source-grounding
operations through a Python MCP server.

The MCP server does not implement Agentic RAG by itself. Agentic RAG behavior
lives outside the MCP server, in Codex/Codex SDK. Codex decides which retrieval
tools to call, whether the returned context is sufficient, and how to synthesize
or act on the returned context.

This is not a single "document QA RAG" server. It targets multiple RAG patterns
that work together:

- Keyword RAG
- Dense Vector RAG
- Hybrid RAG
- Rerank RAG
- Metadata-filtered RAG
- Source-grounded RAG
- GraphRAG
- Temporal RAG
- Trace RAG
- Decision RAG
- Failure/Error RAG
- Code RAG
- Tool RAG
- Memory RAG
- Agentic RAG, enabled by Codex using this MCP tool surface
- Later: Multimodal RAG

## Non-goals

- Do not hide Codex inside MCP. Codex is the MCP client/agent that calls the
  Python MCP server.
- Do not expose a raw vector database MCP as the main product interface.
- Do not make GraphRAG the only retrieval path.
- Do not require paid embedding or reranking APIs for the local retrieval path.
- Do not make the Python server responsible for high-level agent reasoning.

## High-level System Shape

```text
Python application / orchestrator
Codex App / Codex Agent
  └─ calls MCP tools
      ↓
Python MCP server
  ├─ rag_ingest_path / rag_search / rag_get_source
  ├─ graph_search_entities / graph_search_claims
  ├─ graph_expand / graph_get_sources
  ├─ trace_search / decision_search / error_search
  ├─ code_search / tool_search
  └─ memory_store_trace / memory_store_decision / memory_store_failure
      ↓
Local retrieval and storage layer
  ├─ keyword index
  ├─ vector index
  ├─ reranker
  ├─ graph store
  ├─ provenance store
  ├─ trace store
  └─ decision/error/code/tool memory
```

## Role Boundaries

### Codex / Codex SDK

Codex is the agentic controller when it uses this MCP tool surface. Codex SDK is
used only when this project needs Python-managed, non-interactive Codex jobs,
such as graph extraction or graph curation.

Interactive Codex responsibilities:

- Decompose user queries into retrieval steps.
- Decide whether to search keyword/vector/graph/trace/decision/error/code/tool
  memory.
- Judge whether retrieved context is sufficient.
- Request more retrieval through MCP when context is weak.
- Produce final answers, implementation plans, or code changes.

Python-managed Codex SDK job responsibilities:

- Extract entities, relations, claims, aliases, and temporal facts.
- Normalize and canonicalize graph updates.
- Propose graph update plans.
- Support graph curation and evaluation workflows.

### Python MCP Server

The MCP server exposes stable retrieval and memory tools to Codex.

Responsibilities:

- Provide MCP tools with Codex-friendly names and schemas.
- Execute keyword search, vector search, graph search, trace search, and rerank.
- Return compact, source-grounded retrieval results.
- Persist documents, chunks, traces, decisions, failures, tool descriptions, and
  graph structures.
- Validate structured graph extraction outputs before storage.

### Storage/Search Layer

The storage layer owns persistence and indexing.

Implemented default (fully local, server-free):

- Document/chunk/provenance/trace/decision metadata: SQLite.
- Keyword search: SQLite FTS5 (same database file).
- Vector search: Qdrant embedded local mode.
- Graph store: Kuzu.

Postgres and pgvector are selectable behind the same interfaces but are
unimplemented placeholders. See [storage.md](storage.md) for schemas and
adapter boundaries.

The MCP tool contract should not expose the chosen backend directly.

## Target RAG Methods

### Keyword RAG

Keyword search is mandatory for exact terms:

- Error messages
- Function names
- File paths
- Class names
- CLI commands
- Package names
- Issue and PR identifiers

Implementation candidates:

- SQLite FTS5
- Postgres full-text search
- BM25 library, if needed later

### Dense Vector RAG

Vector search is mandatory for semantic recall across docs, traces, decisions,
and source context.

Local embedding candidates:

- `intfloat/multilingual-e5-small` for the first local baseline.
- `intfloat/multilingual-e5-base` for stronger local quality.
- `BAAI/bge-m3` for heavier multilingual retrieval.
- `cl-nagoya/ruri-large` if Japanese-only quality becomes more important.

E5-style models should use query/document prefixes consistently:

```text
query: ...
passage: ...
```

### Hybrid RAG

Hybrid retrieval is mandatory. It combines keyword and vector results.

Baseline algorithm:

```text
keyword_search(query, top_n=50)
vector_search(query, top_n=50)
merge_with_rrf(keyword_results, vector_results)
metadata_filter(...)
return top_n candidates
```

RRF is the default merge strategy because it is simple, stable, and backend
agnostic.

### Rerank RAG

Reranking is mandatory for quality-sensitive retrieval.

Pipeline:

```text
hybrid retrieval top 30-100
→ local cross-encoder reranker
→ top 5-15 context results
```

Local reranker candidates:

- `hotchpotch/japanese-reranker-cross-encoder-small-v1` for Japanese-heavy use.
- `hotchpotch/japanese-reranker-cross-encoder-xsmall-v1` for faster CPU runs.
- `BAAI/bge-reranker-v2-m3` for multilingual quality.
- `Alibaba-NLP/gte-multilingual-reranker-base` as another multilingual option.

Rerank should be configurable per tool call:

```text
rerank: true | false   # omitted -> [reranker].enabled_default
top_k: integer         # results returned after reranking
```

The rerank candidate pool is capped by the `[reranker].max_candidates`
configuration value, not per call.

### Metadata-filtered RAG

All retrieval paths must support metadata filtering.

Common filters:

- `scope`
- `kind`
- `repo`
- `path`
- `language`
- `tags`
- `created_at`
- `updated_at`
- `valid_from`
- `valid_to`
- `source_type`
- `confidence_min`

Common `kind` values:

- `doc`
- `chunk`
- `trace`
- `decision`
- `error`
- `code`
- `tool`
- `entity`
- `relation`
- `claim`
- `community`

`doc`, `trace`, `decision`, `error`, `code`, and `tool` are chunk-backed kinds
served by `rag_search` and the kind-scoped `*_search` tools. `entity`,
`relation`, `claim`, and `community` are graph item kinds served by the
`graph_*` tools; `rag_search` does not return them directly (except via its
`graph_expand` enrichment).

### Source-grounded RAG

Every retrieval result must be traceable to a source.

Required result fields:

```json
{
  "id": "chunk_or_graph_item_id",
  "kind": "doc|trace|decision|error|code|tool|entity|relation|claim",
  "score": 0.0,
  "source_id": "source_id",
  "source_path": "/absolute/or/repo/relative/path",
  "title": "short title",
  "excerpt": "compact excerpt",
  "metadata": {}
}
```

Graph nodes, graph edges, and claims must also retain provenance.

### GraphRAG

GraphRAG is mandatory and first-class. It is not a later optional layer.

Graph model:

- Entities
- Relations
- Claims
- Aliases
- Communities
- Community summaries
- Provenance
- Temporal validity

Core graph operations:

- Entity search
- Claim search
- Relation search
- Neighborhood expansion
- Path finding, later
- Community retrieval (Python service only; no MCP tool yet)
- Source lookup for graph items

Entities and claims are text-searchable: they are indexed into a dedicated
keyword index and into the shared vector collection (distinguished by
`item_kind`), so `graph_search_entities` and `graph_search_claims` run the
same hybrid keyword + vector + RRF pipeline as document search.

Community detection is not implemented yet; the community layer only stores
and retrieves externally provided communities and summaries.

Graph extraction is performed by Codex SDK, not by local heuristic-only logic.
The Python layer validates and persists the extracted structures.

### Temporal RAG

Temporal metadata is mandatory because project facts change.

Required temporal fields where applicable:

- `created_at`
- `updated_at`
- `valid_from`
- `valid_to`
- `supersedes`
- `superseded_by`

Temporal retrieval should prefer current facts by default while allowing older
facts to be inspected when needed.

### Trace RAG

Trace RAG stores and retrieves previous agent/task execution history.

Trace record fields:

- Task
- Prompt or request summary
- Tools used
- Commands run
- Files read
- Files changed
- Errors encountered
- Tests run
- Outcome
- Final response
- Human feedback
- Linked decisions
- Linked graph entities

### Decision RAG

Decision RAG stores durable technical choices and their rationale.

Decision record fields:

- Decision title
- Status: proposed, accepted, rejected, superseded
- Context
- Chosen option
- Rationale
- Alternatives considered (including rejected options)
- Consequences
- Source links
- Related entities
- Supersedes/superseded_by

Storing a decision with `supersedes` marks the old decision superseded and
expires its search chunks, so temporal filtering hides it from default
searches.

### Failure/Error RAG

Failure RAG stores error signatures and resolution history. Records are
written via `memory_store_failure` and retrieved via `error_search`
(chunk kind `error`); "failure" names the record, "error" names the search
surface.

Failure record fields:

- Error text
- Error category
- Command or tool that failed
- Environment
- Suspected cause
- Confirmed cause
- Fix applied
- Verification
- Related traces
- Related code paths

### Code RAG

Code RAG supports source-level retrieval.

Initial scope:

- File path chunks
- Function/class/module metadata when available
- Imports and dependencies
- Test files and fixtures
- Error-to-file associations

Later scope:

- Symbol graph
- Call graph
- AST-derived relationships
- Language-server backed symbol extraction

### Tool RAG

Tool RAG helps Codex choose tools.

Tool records:

- MCP server
- Tool name
- Tool description
- Input schema
- Output shape
- Approval policy
- Rate limits
- Examples
- Known failure modes

This is important when the number of available MCP tools grows.

Tool records are written through the Python API (`ToolMemoryService.store`,
upserting on server + tool name); the MCP surface exposes only `tool_search`.
A CLI registration command may be added later.

### Memory RAG

Memory RAG is the long-term project memory layer. It unifies trace, decision,
failure, tool, code, and graph memory under one retrieval contract: every
memory record is persisted twice — as a typed table row (authoritative) and
as a rendered markdown document indexed through the same chunk/keyword/vector
pipeline as ordinary documents (`source_type = "memory"`,
`source_uri = "memory://{kind}/{record_id}"`). The kind-scoped `*_search`
tools are `rag_search` constrained to a single `kind`.

### Agentic RAG

Agentic RAG is not implemented inside the Python MCP server. It is enabled by
Codex using this MCP tool surface.

Codex controls:

- Query rewriting
- Query decomposition
- Search route selection
- Graph traversal planning
- Re-searching when context is weak
- Choosing whether to call `rag_search`, `graph_expand`, `trace_search`,
  `decision_search`, `error_search`, or `tool_search`
- Final context sufficiency judgment

The Python MCP server provides deterministic retrieval, memory, graph, rerank,
and source lookup tools; Codex decides how to use them.

### Multimodal RAG

Multimodal RAG is a later target.

Likely additions:

- PDF layout extraction
- Tables
- Images
- Diagrams
- Screenshots

Potential parser:

- Docling

## MCP Tool Surface

The tool surface is fixed at these 15 tools:

```text
rag_ingest_path(path, scope, tags, options)
rag_search(query, scopes, kinds, filters, top_k, rerank, graph_expand)
rag_get_source(source_id, around)

graph_search_entities(query, filters, top_k)
graph_search_claims(query, filters, top_k)
graph_expand(entity_id, depth, relation_types, filters)
graph_get_sources(graph_item_id)

trace_search(query, filters, top_k, rerank)
decision_search(query, filters, top_k, rerank)
error_search(query, filters, top_k, rerank)
code_search(query, filters, top_k, rerank)
tool_search(query, filters, top_k, rerank)

memory_store_trace(payload)
memory_store_decision(payload)
memory_store_failure(payload)
```

Semantics that follow from the result shape:

- Search results carry `id` (chunk or graph item id) and `source_id`
  (document id). `rag_get_source` accepts either: a chunk id returns that
  chunk with `around` neighboring chunks; a document id returns the head of
  the document.
- `graph_expand=true` on `rag_search` appends up to 5 graph entities linked
  (via provenance) to the returned chunks, as `kind="entity"` results at the
  tail.
- The kind-scoped searches (`trace/decision/error/code/tool_search`) are
  `rag_search` constrained to one chunk kind.

Tool outputs must be compact. Each search result should include enough context
for the agent to decide whether to fetch more via `rag_get_source` or
`graph_get_sources`.

Full parameter and response contracts live in [mcp-tools.md](mcp-tools.md).

## Data Model

### Documents

```text
documents
  id
  source_type          -- file | memory | graph
  source_uri
  title
  content_hash
  scope
  kind
  repo
  language
  tags
  metadata
  created_at
  updated_at
  -- unique (source_uri, scope)
```

### Chunks

```text
chunks
  id
  document_id
  ordinal
  heading_path
  text
  token_count
  -- denormalized filter columns copied from the document:
  scope
  kind
  repo
  path
  language
  tags
  source_type
  valid_from
  valid_to
  metadata
  embedding_id         -- vector index point id
  created_at
  updated_at
```

### Entities

```text
entities
  id
  canonical_name
  entity_type
  description
  metadata
  created_at
  updated_at
```

### Aliases

```text
entity_aliases
  id
  entity_id
  alias
  normalized_alias     -- NFKC + casefold + collapsed whitespace; lookup key
  source_id
  confidence
```

### Relations

```text
relations
  id
  source_entity_id
  target_entity_id
  relation_type
  description
  confidence
  valid_from
  valid_to
  metadata
  created_at
```

### Claims

```text
claims
  id
  subject_entity_id
  predicate
  object_text
  object_entity_id
  modality
  confidence
  valid_from
  valid_to
  metadata
  created_at
```

### Communities

```text
communities
  id
  title
  summary
  level
  created_at
  -- plus entity membership edges in the graph store
```

### Provenance

```text
provenance
  id
  item_type
  item_id
  source_id
  chunk_id
  evidence_text
  char_start
  char_end
  extraction_run_id
```

### Extraction Runs

```text
extraction_runs
  id
  source_id
  chunk_id
  codex_thread_id
  prompt_version
  status
  started_at
  completed_at
  error
```

### Traces

```text
traces
  id
  task
  outcome
  tools_used
  commands
  files_read
  files_changed
  errors
  tests
  final_response
  human_feedback
  linked_decisions
  linked_entities
  metadata             -- carries prompt/request summary when provided
  created_at
```

### Decisions

```text
decisions
  id
  title
  status
  context
  decision
  rationale
  alternatives
  consequences
  source_links
  related_entities
  supersedes
  superseded_by
  metadata
  created_at
  updated_at
```

### Failures

```text
failures
  id
  error_text
  error_category
  command
  environment
  suspected_cause
  confirmed_cause
  fix_applied
  verification
  related_traces
  related_code_paths
  metadata
  created_at
```

### Tool Records

```text
tool_entries
  id
  server
  name
  description
  input_schema
  output_shape
  approval_policy
  rate_limits
  examples
  known_failure_modes
  metadata
  created_at
  updated_at
  -- unique (server, name); upserted
```

## Graph Extraction with Codex SDK

Extraction should use a fixed JSON schema.

Required extraction output:

```json
{
  "entities": [
    {
      "name": "string",
      "type": "string",
      "aliases": ["string"],
      "description": "string",
      "confidence": 0.0,
      "evidence": "string"
    }
  ],
  "relations": [
    {
      "source": "string",
      "target": "string",
      "type": "string",
      "description": "string",
      "confidence": 0.0,
      "evidence": "string",
      "valid_from": null,
      "valid_to": null
    }
  ],
  "claims": [
    {
      "subject": "string",
      "predicate": "string",
      "object": "string",
      "modality": "fact|hypothesis|decision|requirement|constraint",
      "confidence": 0.0,
      "evidence": "string",
      "valid_from": null,
      "valid_to": null
    }
  ]
}
```

Validation steps:

```text
Codex extraction
→ JSON parse
→ schema validation
→ evidence presence check
→ canonicalization
→ duplicate detection
→ graph update plan
→ persistence
```

Dispatch timing: ingestion queues per-chunk extraction runs
(`status = "pending"`) only when requested (`options.extract` on
`rag_ingest_path`, or `codex.auto_extract_on_ingest`, default off; queued only
for kinds in `codex.extract_kinds`, default `["doc"]`). Pending runs are
drained by the `extract` CLI command or an explicit orchestrator call — Codex
SDK is never invoked inside an MCP tool call.

## Retrieval Flow

Default retrieval flow:

```text
Codex receives task
→ Codex plans retrieval strategy
→ Codex calls MCP tools
→ Python server runs keyword/vector/graph/trace retrieval
→ Python server optionally reranks
→ Python server returns compact source-grounded results
→ Codex judges sufficiency
→ Codex asks for more retrieval if needed
→ Codex synthesizes answer or acts
→ Python stores useful trace/decision/failure memory
```

## Local Model Policy

Retrieval infrastructure should be able to run without paid API calls.

Local by default:

- Embeddings
- Reranking
- Keyword search
- Vector search
- Graph search
- Metadata filtering

Codex is intentionally used for agentic reasoning. Codex SDK is used for
non-interactive Codex-driven jobs such as graph extraction and curation.

## Configuration Sketch

Defaults are the fully local configuration. TOML file
(`./multi-rag-harness.toml`, `--config`, or `$MRH_CONFIG_FILE`) with
`MRH_`-prefixed environment variable overrides (`__` for nesting).

```toml
data_dir = "./.multi-rag-harness"

[embedding]
model = "intfloat/multilingual-e5-base"
device = "auto"
dimension = 768

[reranker]
model = "hotchpotch/japanese-reranker-cross-encoder-small-v1"
enabled_default = true
max_candidates = 50

[storage]
metadata_backend = "sqlite"    # "postgres" selectable, not implemented
vector_backend = "qdrant"      # embedded local mode; "pgvector" not implemented
graph_backend = "kuzu"

[mcp]
server_name = "multi-rag-harness"

[codex]
prompt_version = "extraction/v1"
auto_extract_on_ingest = false
extract_kinds = ["doc"]
max_runs_per_batch = 25
```

## Resolved Design Decisions

- Graph backend: Kuzu (entities, relations, community membership); claims,
  aliases, and provenance stay in the metadata store.
- Vector storage: Qdrant embedded local mode (server-free); pgvector remains a
  selectable placeholder.
- Reranking: enabled by default for all searches
  (`[reranker].enabled_default = true`), overridable per tool call.
- Codex extraction batching: pending runs queued at ingest (opt-in), drained
  in batches (`codex.max_runs_per_batch`) by the `extract` CLI command.

## Open Design Questions

- How should Codex extraction results be cached across re-runs of unchanged
  chunks?
- How should source spans be represented for non-text sources later?
- What is the first benchmark set for retrieval quality?
- When should community detection be implemented, and with which algorithm?
