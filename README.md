# multi-rag-harness

Multi-RAG retrieval and memory substrate for Codex agents. It provides the MCP
tool surface for retrieval, memory, graph exploration, reranking, and
source-grounded context. Agentic RAG behavior lives in Codex, which calls these
tools.

See [docs/specification.md](docs/specification.md) for the design,
[docs/mcp-tools.md](docs/mcp-tools.md) for the tool contracts, and
[docs/storage.md](docs/storage.md) for the storage layer.

## Usage

The default configuration is fully local (SQLite + FTS5, embedded Qdrant,
Kuzu) and stores data under `./.multi-rag-harness`. Local embedding and
reranker models download from Hugging Face on first use.

```bash
uv run multi-rag-harness serve                 # MCP server over stdio
uv run multi-rag-harness ingest PATH           # index docs/code
uv run multi-rag-harness extract               # run pending Codex graph extraction
uv run multi-rag-harness config-show           # resolved configuration
```

Configuration: `multi-rag-harness.toml` (or `--config` / `$MRH_CONFIG_FILE`),
overridable with `MRH_`-prefixed env vars (e.g. `MRH_EMBEDDING__MODEL`).

## Development

This repository uses `uv` for Python packaging and dependency management.

```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
uv run ty check
```

Tests run entirely offline: fake embedding/reranker/Codex clients, tmp-dir
backends, no model downloads.
