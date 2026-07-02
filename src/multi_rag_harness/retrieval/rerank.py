"""Rerank orchestration."""

from __future__ import annotations

from collections.abc import Sequence

from multi_rag_harness.models.reranker import Reranker
from multi_rag_harness.retrieval.results import RetrievalCandidate


class RerankService:
    def __init__(self, reranker: Reranker) -> None:
        self._reranker = reranker

    async def rerank(
        self, query: str, candidates: Sequence[RetrievalCandidate], top_k: int
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []
        scores = await self._reranker.score(query, [c.chunk.text for c in candidates])
        rescored = [
            candidate.model_copy(update={"rerank_score": score})
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda c: (-(c.rerank_score or 0.0), -c.rrf_score))
        return rescored[:top_k]
