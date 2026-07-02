# AGENTS.md

## Project Purpose

`multi-rag-harness` is a Python-based, local-first retrieval and memory
substrate for Codex agents. It is not a single document-QA RAG server, and it
does not implement Agentic RAG inside the MCP server.

The project targets these RAG methods as first-class concerns:

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
- Agentic RAG enabled by Codex using this MCP tool surface
- Later: Multimodal RAG

Use `docs/specification.md` as the source of truth for project scope.
Update it when implementation decisions change the system shape, tool surface,
storage model, or supported RAG methods.

## Boundary Rules

- Codex is not inside MCP. Codex is the agent/MCP client that calls this
  project's Python MCP server.
- Agentic RAG behavior lives outside the MCP server, in Codex/Codex SDK.
  Codex owns query decomposition, traversal planning, sufficiency checks, and
  final synthesis.
- Codex SDK may also run non-interactive graph extraction and curation jobs.
- The Python MCP server owns stable tool contracts, persistence, search,
  graph traversal, reranking execution, validation, and compact source-grounded
  responses.
- Do not expose raw backend-specific primitives as the primary public interface.
  Keep Qdrant, pgvector, Kuzu, SQLite, or Postgres behind project-level tools.
- Do not reduce the project to GraphRAG only. GraphRAG is mandatory, but it must
  compose with hybrid retrieval, reranking, trace memory, decision memory,
  failure memory, code retrieval, and tool retrieval.
- All retrieved items, graph nodes, graph edges, and claims must retain
  provenance to source documents, chunks, traces, or decisions.

## Python And Dependency Management

- Use `uv` for Python package and dependency management.
- Do not use `pip install`, Poetry, or ad hoc virtualenv commands for project
  dependencies unless the repository intentionally changes package management.
- Add runtime dependencies with `uv add`.
- Add development-only dependencies with `uv add --dev`.
- Keep `uv.lock` committed and in sync with `pyproject.toml`.
- Prefer Python 3.11+ unless `pyproject.toml` is intentionally changed.

## Local Models

Default local model direction:

- Embedding baseline: `intfloat/multilingual-e5-base`
- Lighter embedding option: `intfloat/multilingual-e5-small`
- Heavier multilingual embedding option: `BAAI/bge-m3`
- Japanese reranker baseline: `hotchpotch/japanese-reranker-cross-encoder-small-v1`
- Multilingual reranker option: `BAAI/bge-reranker-v2-m3`

Do not introduce paid embedding or reranking APIs as required dependencies for
the default local retrieval path.

## Implementation Expectations

- Keep modules small and aligned to boundaries such as MCP tools, retrieval,
  graph, storage, models, Codex SDK orchestration, and configuration.
- Prefer typed Pydantic models for MCP inputs/outputs, extraction results,
  search results, graph updates, and persisted records.
- Validate Codex SDK graph extraction output before storage.
- Keep tool outputs compact enough for agent use. Provide `get` or source
  expansion tools for full context.
- Prefer backend-neutral interfaces before writing backend-specific adapters.
- When adding a backend adapter, keep it isolated behind a protocol or service
  boundary.
- Treat trace, decision, error, code, tool, and graph memory as durable data,
  not transient prompt text.

## Verification Commands

Run the smallest relevant set after changes:

```bash
uv lock --check
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
uv build --sdist --wheel --out-dir /tmp/multi-rag-harness-dist
```

For dependency-only changes, at minimum run:

```bash
uv lock --check
uv build --sdist --wheel --out-dir /tmp/multi-rag-harness-dist
```

If a command cannot run locally because a dependency is too heavy, missing, or
requires external state, report the exact command and the reason.

## Git And Documentation

- Keep changes scoped to the requested task.
- Do not rewrite existing specifications casually. Update docs deliberately when
  behavior, project structure, or system shape changes.
- Commit generated lockfile changes when dependencies change.
- Avoid committing local caches, virtual environments, model weights, database
  files, or build outputs.
- Use concise commit messages that describe the durable project change.
