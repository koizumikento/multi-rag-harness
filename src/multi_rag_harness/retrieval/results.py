"""Retrieval result models and helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from multi_rag_harness.storage.interfaces import ChunkRecord

EXCERPT_MAX_CHARS = 480


def make_excerpt(text: str, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


class SearchResult(BaseModel):
    """Compact source-grounded result; field set fixed by the specification."""

    id: str
    kind: str
    score: float
    source_id: str
    source_path: str
    title: str
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalCandidate(BaseModel):
    """Internal pre-shaping candidate carrying ranking scores."""

    chunk: ChunkRecord
    rrf_score: float
    rerank_score: float | None = None


class SearchOutput(BaseModel):
    results: list[SearchResult]
    reranked: bool
