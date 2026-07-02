# OSS Release Checklist

Use this checklist before making the repository public or publishing a package.

## Required Owner Decisions

- Confirm the MIT license is the intended public license.
- Confirm the public repository URL in `pyproject.toml` and `README.md`.
- Decide whether the package will be published to PyPI, distributed as a Codex
  plugin, or both.
- Configure private vulnerability reporting or another private security contact
  path.

## Repository Hygiene

- Run a secret scan before publishing.
- Confirm `.multi-rag-harness/`, `.venv/`, caches, logs, model weights, and
  local database files are ignored.
- Confirm plugin metadata does not contain machine-local absolute paths.
- Confirm examples do not contain real API keys, private paths, or proprietary
  data.

## Verification

```bash
uv lock --check
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
uv build --sdist --wheel --out-dir /tmp/multi-rag-harness-dist
```

## Documentation

- README explains install, usage, configuration, local model behavior, and MCP
  setup.
- `docs/specification.md` matches the implemented tool surface and storage
  model.
- `docs/mcp-tools.md` describes current tool contracts.
- `docs/storage.md` documents default local data paths and backend boundaries.
- `CONTRIBUTING.md` and `SECURITY.md` are present.
