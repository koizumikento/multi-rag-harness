# Contributing

Thanks for helping improve `multi-rag-harness`.

## Development Setup

Use `uv` for all Python dependency management.

```bash
uv sync
uv run multi-rag-harness config-show
```

Use `uv sync --extra codex` when working on Codex SDK extraction behavior.

Do not install project dependencies with `pip`, Poetry, or ad hoc virtualenv
commands unless the repository intentionally changes package management.

## Project Boundaries

- Codex is the MCP client and agentic controller; it is not embedded inside the
  MCP server.
- The Python MCP server owns stable tool contracts, persistence, retrieval,
  graph traversal, reranking execution, validation, and compact
  source-grounded responses.
- Keep backend-specific details behind project-level interfaces.
- Preserve provenance for retrieved items, graph data, claims, traces,
  decisions, failures, code, and tool memory.
- Do not make paid embedding or reranking APIs required for the default local
  retrieval path.

The durable scope reference is [docs/specification.md](docs/specification.md).
Update it when implementation decisions change the system shape, tool surface,
storage model, or supported RAG methods.

## Checks

Run the smallest relevant set for your change. Before opening a PR that changes
runtime code, run:

```bash
uv lock --check
uv run ruff check
uv run ruff format --check
uv run ty check
uv run coverage run -m pytest
uv run coverage report
uv build --sdist --wheel --out-dir /tmp/multi-rag-harness-dist
```

For dependency-only changes, run at minimum:

```bash
uv lock --check
uv build --sdist --wheel --out-dir /tmp/multi-rag-harness-dist
```

If a check cannot run locally because a dependency is too heavy, missing, or
requires external state, include the exact command and reason in the PR.

## Pull Requests

- Keep changes scoped to one durable project change.
- Include tests for behavior changes.
- Update docs when behavior, project structure, or tool contracts change.
- Commit `uv.lock` with dependency changes.
- Do not commit local caches, virtual environments, model weights, database
  files, build outputs, logs, secrets, or personal configuration.
