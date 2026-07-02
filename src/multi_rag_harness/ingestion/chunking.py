"""Document chunking strategies.

Token counts are estimated without a tokenizer: one token per CJK/fullwidth
codepoint plus roughly one token per four remaining characters. Good enough
for packing chunks to a size budget.
"""

from __future__ import annotations

import math
import re
import unicodedata

from pydantic import BaseModel

DEFAULT_MAX_TOKENS = 400
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class ChunkDraft(BaseModel):
    ordinal: int
    heading_path: str
    text: str
    token_count: int


def estimate_tokens(text: str) -> int:
    wide = sum(1 for ch in text if unicodedata.east_asian_width(ch) in ("W", "F"))
    other = len(text) - wide
    return wide + math.ceil(other / 4)


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip("\n") for p in re.split(r"\n\s*\n", text)]
    return [p for p in paragraphs if p.strip()]


def _split_oversized(paragraph: str, max_tokens: int) -> list[str]:
    """Split a paragraph exceeding the budget at line boundaries. A single
    line larger than the budget is kept whole."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in paragraph.splitlines():
        line_tokens = estimate_tokens(line)
        if current and current_tokens + line_tokens > max_tokens:
            pieces.append("\n".join(current))
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += line_tokens
    if current:
        pieces.append("\n".join(current))
    return pieces


def _pack_paragraphs(
    paragraphs: list[str], heading_path: str, max_tokens: int, start_ordinal: int
) -> list[ChunkDraft]:
    units: list[str] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) > max_tokens:
            units.extend(_split_oversized(paragraph, max_tokens))
        else:
            units.append(paragraph)

    drafts: list[ChunkDraft] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(current)
        drafts.append(
            ChunkDraft(
                ordinal=start_ordinal + len(drafts),
                heading_path=heading_path,
                text=text,
                token_count=estimate_tokens(text),
            )
        )
        current = []
        current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            flush()
        current.append(unit)
        current_tokens += unit_tokens
    flush()
    return drafts


def chunk_text(text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[ChunkDraft]:
    """Plain-text chunking: paragraph packing without heading awareness."""
    return _pack_paragraphs(_split_paragraphs(text), "", max_tokens, 0)


def chunk_markdown(text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[ChunkDraft]:
    """Heading-aware markdown chunking.

    ATX headings maintain a stack producing ``heading_path`` values such as
    ``"Setup > Install"``. Section bodies are paragraph-packed up to
    ``max_tokens``; oversized paragraphs split at line boundaries.
    """
    sections: list[tuple[str, list[str]]] = []  # (heading_path, lines)
    stack: list[tuple[int, str]] = []  # (level, title)
    current_lines: list[str] = []

    def heading_path() -> str:
        return " > ".join(title for _, title in stack)

    def close_section() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines)
        if body.strip():
            sections.append((heading_path(), current_lines))
        current_lines = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            close_section()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        else:
            current_lines.append(line)
    close_section()

    drafts: list[ChunkDraft] = []
    for path, lines in sections:
        paragraphs = _split_paragraphs("\n".join(lines))
        drafts.extend(_pack_paragraphs(paragraphs, path, max_tokens, len(drafts)))
    return drafts


def chunk_code(text: str, max_lines: int = 120, overlap: int = 20) -> list[ChunkDraft]:
    """Sliding line-window chunking for source code."""
    lines = text.splitlines()
    if not lines:
        return []
    if overlap >= max_lines:
        raise ValueError("overlap must be smaller than max_lines")
    drafts: list[ChunkDraft] = []
    step = max_lines - overlap
    start = 0
    while start < len(lines):
        window = lines[start : start + max_lines]
        body = "\n".join(window)
        if body.strip():
            drafts.append(
                ChunkDraft(
                    ordinal=len(drafts),
                    heading_path="",
                    text=body,
                    token_count=estimate_tokens(body),
                )
            )
        if start + max_lines >= len(lines):
            break
        start += step
    return drafts
