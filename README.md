# multi-rag-harness

Multi-RAG retrieval and memory substrate for Codex agents. It provides the MCP
tool surface for retrieval, memory, graph exploration, reranking, and
source-grounded context. Agentic RAG behavior lives in Codex, which calls these
tools.

See [docs/specification.md](docs/specification.md) for the initial design.

## Development

This repository uses `uv` for Python packaging and dependency management.

```bash
uv sync
uv run pytest
uv run ruff check
uv run ty check
```
