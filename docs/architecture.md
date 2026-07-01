# Architecture

This document describes the repository layout, package responsibilities, and
dependency direction for `multi-rag-harness`.

For product scope and RAG method coverage, see
[specification.md](specification.md).

## Repository Layout

```text
multi-rag-harness/
├── AGENTS.md
├── README.md
├── docs/
│   ├── architecture.md
│   ├── mcp-tools.md
│   ├── specification.md
│   └── storage.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── multi_rag_harness/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── codex/
│       ├── graph/
│       ├── ingestion/
│       ├── mcp_server/
│       ├── memory/
│       ├── models/
│       ├── retrieval/
│       └── storage/
└── tests/
    ├── integration/
    └── unit/
```

## Top-level Files

### `AGENTS.md`

Persistent project guidance for Codex. Keep durable rules here: architecture
boundaries, dependency management, verification commands, and implementation
expectations.

### `README.md`

Short project introduction and development commands. Keep this concise.

### `docs/`

Project design documentation.

- `specification.md`: source of truth for scope, supported RAG methods, MCP
  tools, data model, and retrieval flow.
- `architecture.md`: repository layout, package boundaries, and dependency
  direction.
- `mcp-tools.md`: detailed MCP tool contracts once implementation starts.
- `storage.md`: backend schemas, adapter boundaries, and migration notes.

### `pyproject.toml` and `uv.lock`

Python package metadata and locked dependencies. Use `uv` for all dependency
changes.

## Package Responsibilities

### `multi_rag_harness.cli`

Command-line entry points. This package should stay thin and delegate to
application services.

Examples:

- start the MCP server
- run ingestion jobs
- run extraction jobs
- inspect configuration

### `multi_rag_harness.config`

Configuration loading and typed settings.

Owns:

- environment-driven settings
- local paths
- backend selection
- model names
- MCP server settings
- Codex SDK extraction settings

### `multi_rag_harness.mcp_server`

The Python MCP server boundary. This is the tool surface Codex calls.

Owns:

- server assembly
- tool registration
- MCP input/output schemas
- thin tool handlers
- response shaping for agent consumption

It should not contain backend-specific storage logic. It should call retrieval,
graph, memory, and storage services through internal interfaces.

### `multi_rag_harness.codex`

Codex SDK orchestration.

Owns:

- Codex SDK client helpers
- graph extraction jobs
- canonicalization prompts
- query planning prompts
- extraction prompt versions
- structured extraction result handling before validation

Codex SDK is the agentic controller. It owns query decomposition, extraction,
graph update planning, traversal planning, sufficiency judgment, and final
synthesis. The Python code should orchestrate those jobs and validate their
outputs.

### `multi_rag_harness.retrieval`

Retrieval algorithms and result composition.

Owns:

- keyword retrieval
- vector retrieval
- hybrid retrieval
- RRF or other rank fusion
- rerank orchestration
- retrieval result models
- metadata-filtered search behavior

It should depend on storage interfaces and model adapters, not on concrete MCP
tool handlers.

### `multi_rag_harness.graph`

GraphRAG domain behavior.

Owns:

- entity and relation models
- claim models
- graph extraction result normalization
- entity canonicalization
- graph traversal
- neighborhood expansion
- community summaries

It should preserve provenance for every graph item. It should not directly call
Codex SDK unless the call is part of a clearly isolated extraction workflow in
`multi_rag_harness.codex`.

### `multi_rag_harness.memory`

Durable project memory beyond ordinary documents.

Owns:

- trace memory
- decision memory
- failure/error memory
- code memory
- tool memory

These memory types are first-class RAG sources. They should expose typed domain
services and use storage adapters for persistence.

### `multi_rag_harness.ingestion`

Document and source ingestion.

Owns:

- loading documents from paths or future sources
- normalization
- chunking
- content hashing
- ingestion pipeline orchestration
- dispatching extraction jobs after chunks are created

It should not decide final agent behavior. It prepares material for retrieval
and graph extraction.

### `multi_rag_harness.models`

Local embedding and reranker adapters.

Owns:

- embedding model loading
- query/document prefix handling
- vector generation
- reranker model loading
- rerank scoring
- local model cache configuration

Default direction:

- embedding: `intfloat/multilingual-e5-base`
- lighter embedding: `intfloat/multilingual-e5-small`
- heavier embedding: `BAAI/bge-m3`
- Japanese reranker: `hotchpotch/japanese-reranker-cross-encoder-small-v1`
- multilingual reranker: `BAAI/bge-reranker-v2-m3`

### `multi_rag_harness.storage`

Persistence and backend adapters.

Owns:

- storage interfaces
- SQLite adapter
- Postgres adapter
- pgvector adapter
- Qdrant adapter
- Kuzu adapter
- transaction and lifecycle boundaries

The public application layers should not expose backend-specific concepts
unless a feature explicitly requires it.

## Dependency Direction

Keep dependencies flowing inward toward domain services and outward only through
interfaces.

Preferred direction:

```text
cli
  → config
  → mcp_server

mcp_server
  → retrieval
  → graph
  → memory
  → ingestion
  → storage interfaces

codex
  → graph models
  → memory models
  → storage interfaces, when persisting extraction runs

retrieval
  → models
  → storage interfaces

graph
  → storage interfaces

memory
  → storage interfaces

ingestion
  → retrieval models
  → graph extraction handoff
  → storage interfaces

storage
  → external backends
```

Avoid these dependency shapes:

```text
storage → mcp_server
storage → codex
retrieval → mcp_server
graph → mcp_server
memory → mcp_server
models → mcp_server
```

## Where New Code Goes

- New MCP tool schema: `mcp_server/schemas.py`
- New MCP tool handler: `mcp_server/tools.py`
- MCP server wiring: `mcp_server/server.py`
- New Codex SDK extraction workflow: `codex/extraction.py`
- New Codex prompt template: `codex/prompts.py`
- New retrieval strategy: `retrieval/`
- New reranking behavior: `retrieval/rerank.py` or `models/reranker.py`
- New embedding behavior: `models/embedding.py`
- New graph traversal behavior: `graph/traversal.py`
- New entity merge/canonicalization behavior: `graph/canonicalize.py`
- New memory type: `memory/`
- New storage backend: `storage/`
- New ingestion source: `ingestion/documents.py`
- New chunking strategy: `ingestion/chunking.py`
- Cross-cutting configuration: `config.py`
- CLI command: `cli.py`

## Testing Layout

Use `tests/unit/` for pure units:

- schema validation
- rank fusion
- query/document prefix handling
- extraction output validation
- canonicalization helpers
- storage interface contract tests with fakes

Use `tests/integration/` for backend or process boundaries:

- SQLite/Postgres storage behavior
- Qdrant/pgvector vector behavior
- Kuzu graph behavior
- MCP server tool calls
- Codex SDK orchestration tests with controlled fixtures or fakes

Avoid integration tests that require model downloads or live Codex calls by
default. Gate expensive or external-state tests explicitly.

## Documentation Rules

- Update `specification.md` when scope, supported RAG methods, tool surface, or
  data model changes.
- Update `architecture.md` when package responsibilities or dependency
  direction changes.
- Update `mcp-tools.md` when MCP tool schemas or semantics become concrete.
- Update `storage.md` when backend schemas, adapters, or migration strategy
  become concrete.

