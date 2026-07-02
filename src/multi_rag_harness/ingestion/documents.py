"""Document loading and normalization."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

MARKDOWN_EXTS = {".md", ".markdown"}
TEXT_EXTS = {".txt", ".rst"}
CODE_EXTS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "shell",
    ".sql": "sql",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".multi-rag-harness",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
}
MAX_FILE_BYTES = 2 * 1024 * 1024

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class LoadedDocument(BaseModel):
    source_uri: str  # absolute path
    title: str
    text: str
    content_hash: str
    kind: str  # doc | code
    language: str | None = None
    path: str  # cwd-relative when possible, else absolute


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _display_path(file: Path) -> str:
    resolved = file.resolve()
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


def _load_file(file: Path) -> LoadedDocument | None:
    suffix = file.suffix.lower()
    if suffix in MARKDOWN_EXTS:
        kind, language = "doc", None
    elif suffix in TEXT_EXTS:
        kind, language = "doc", None
    elif suffix in CODE_EXTS:
        kind, language = "code", CODE_EXTS[suffix]
    else:
        return None
    if file.stat().st_size > MAX_FILE_BYTES:
        return None
    text = file.read_text(encoding="utf-8", errors="replace")
    title = file.name
    if suffix in MARKDOWN_EXTS:
        match = _H1_RE.search(text)
        if match:
            title = match.group(1).strip()
    return LoadedDocument(
        source_uri=str(file.resolve()),
        title=title,
        text=text,
        content_hash=_content_hash(text),
        kind=kind,
        language=language,
        path=_display_path(file),
    )


def iter_documents(root: Path) -> Iterator[LoadedDocument]:
    """Yield loadable documents under ``root`` (a file or a directory)."""
    root = root.expanduser()
    if root.is_file():
        loaded = _load_file(root)
        if loaded is not None:
            yield loaded
        return
    if not root.is_dir():
        raise FileNotFoundError(f"ingest path does not exist: {root}")
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        if any(part in SKIP_DIRS for part in file.parts):
            continue
        loaded = _load_file(file)
        if loaded is not None:
            yield loaded
