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

External connections are opt-in via env vars (local stays the default):

```bash
# OpenAI-compatible embeddings endpoint (OpenAI / TEI / vLLM / Ollama)
export MRH_EMBEDDING__PROVIDER=api
export MRH_EMBEDDING__BASE_URL=https://api.openai.com/v1
export MRH_EMBEDDING__MODEL=text-embedding-3-small
export MRH_EMBEDDING__DIMENSION=1536
export MRH_EMBEDDING__API_KEY=sk-...

# Cohere/Jina-compatible rerank endpoint
export MRH_RERANKER__PROVIDER=api
export MRH_RERANKER__BASE_URL=https://api.jina.ai/v1
export MRH_RERANKER__MODEL=jina-reranker-v2-base-multilingual
export MRH_RERANKER__API_KEY=...

# Remote Qdrant server (unset -> embedded local mode)
export MRH_STORAGE__QDRANT_URL=https://qdrant.example.com:6333
export MRH_STORAGE__QDRANT_API_KEY=...
```

Keep API keys in env vars, not in the TOML file. Note: vector collections are
pinned to the embedding dimension — switching embedding models requires
re-ingesting into a fresh `data_dir`.

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
