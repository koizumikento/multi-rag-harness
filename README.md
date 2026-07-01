# multi-rag-harness

Agentic multi-RAG harness for Codex-driven retrieval, memory, graph exploration,
reranking, and MCP tool access.

See [docs/specification.md](docs/specification.md) for the initial design.

## Development

This repository uses `uv` for Python packaging and dependency management.

```bash
uv sync
uv run pytest
uv run ruff check
uv run ty check
```
