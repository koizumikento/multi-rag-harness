# multi-rag-harness Specification

## Purpose

`multi-rag-harness` is a local-first retrieval and memory substrate for agentic
workflows. The primary agent controller is Codex via the Codex SDK. The harness
exposes retrieval and graph operations through a Python MCP server, while Codex
performs the agentic parts: query planning, graph extraction, relation
normalization, traversal planning, sufficiency checks, and final synthesis.

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
- Agentic RAG, controlled by Codex SDK
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
  ├─ Codex SDK
  │   ├─ graph extraction
  │   ├─ entity canonicalization
  │   ├─ relation and claim extraction
  │   ├─ query decomposition
  │   ├─ graph traversal planning
  │   ├─ retrieval sufficiency judgment
  │   └─ final synthesis or implementation planning
  │
  └─ Python MCP server
      ├─ rag_search
      ├─ rag_get_source
      ├─ graph_search_entities
      ├─ graph_expand
      ├─ trace_search
      ├─ decision_search
      ├─ error_search
      ├─ tool_search
      └─ memory_store
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

### Codex SDK

Codex SDK is the agentic controller and graph intelligence layer.

Responsibilities:

- Extract entities, relations, claims, aliases, and temporal facts.
- Normalize and canonicalize graph updates.
- Propose graph update plans.
- Decompose user queries into retrieval steps.
- Decide whether to search keyword/vector/graph/trace/decision/error/code/tool
  memory.
- Judge whether retrieved context is sufficient.
- Request more retrieval through MCP when context is weak.
- Produce final answers, implementation plans, or code changes.

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

Initial backend direction:

- Document/chunk/provenance/trace/decision metadata: Postgres or SQLite.
- Vector search: pgvector, Qdrant, sqlite-vec, or another swappable backend.
- Graph store: Kuzu, Neo4j, or a relational graph schema.
- Keyword search: Postgres full-text search or SQLite FTS5.

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
rerank: true | false
rerank_top_n: integer
```

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
- Community retrieval
- Source lookup for graph items

Graph extraction is performed by Codex SDK, not by local heuristic-only logic.
The Python layer validates and persists the extracted structures.

### Temporal RAG

Temporal metadata is mandatory because project facts change.

Required temporal fields where applicable:

- `created_at`
- `updated_at`
- `observed_at`
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
- Alternatives considered
- Rejected options
- Consequences
- Source links
- Related entities
- Supersedes/superseded_by

### Failure/Error RAG

Failure RAG stores error signatures and resolution history.

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

### Memory RAG

Memory RAG is the long-term project memory layer. It should unify trace,
decision, failure, tool, code, and graph memory under one retrieval contract.

### Agentic RAG

Agentic RAG is implemented by Codex SDK.

Codex controls:

- Query rewriting
- Query decomposition
- Search route selection
- Graph traversal planning
- Re-searching when context is weak
- Choosing whether to call `rag_search`, `graph_expand`, `trace_search`,
  `decision_search`, `error_search`, or `tool_search`
- Final context sufficiency judgment

The Python server provides tools; Codex decides how to use them.

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

Initial tool surface:

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

Tool outputs must be compact. Each search result should include enough context
for the agent to decide whether to fetch more via `rag_get_source` or
`graph_get_sources`.

## Data Model

### Documents

```text
documents
  id
  source_type
  source_uri
  title
  content_hash
  metadata
  created_at
  updated_at
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
  metadata
  embedding_id
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
  metadata
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
  supersedes
  superseded_by
  metadata
  created_at
  updated_at
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

Codex SDK is intentionally used for agentic reasoning and graph extraction.

## Configuration Sketch

```toml
[embedding]
model = "intfloat/multilingual-e5-base"
device = "auto"

[reranker]
model = "hotchpotch/japanese-reranker-cross-encoder-small-v1"
enabled_default = true

[storage]
metadata_backend = "postgres"
vector_backend = "pgvector"
graph_backend = "kuzu"

[mcp]
server_name = "multi-rag-harness"
```

## Open Design Questions

- Should the first graph backend be Kuzu, Neo4j, or Postgres tables?
- Should vector storage start with pgvector, Qdrant, or sqlite-vec?
- Should reranking be enabled by default for all searches or only for complex
  searches?
- How should Codex extraction jobs be batched and cached?
- How should source spans be represented for non-text sources later?
- What is the first benchmark set for retrieval quality?
