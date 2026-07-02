"""Hybrid retrieval: keyword + vector merged with reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence

from multi_rag_harness.config import Settings
from multi_rag_harness.retrieval.keyword import KeywordRetriever
from multi_rag_harness.retrieval.rerank import RerankService
from multi_rag_harness.retrieval.results import (
    RetrievalCandidate,
    SearchOutput,
    SearchResult,
    make_excerpt,
)
from multi_rag_harness.retrieval.vector import VectorRetriever
from multi_rag_harness.storage.interfaces import (
    ChunkRecord,
    DocumentRecord,
    MetadataStore,
    ScoredId,
    SearchFilters,
    utcnow,
)

RRF_K = 60


def rrf_merge(result_lists: Sequence[Sequence[ScoredId]], k: int = RRF_K) -> list[ScoredId]:
    """Reciprocal rank fusion: ``score(d) = sum over lists of 1 / (k + rank)``,
    rank is 1-based. Ties break by id ascending for determinism."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, scored in enumerate(results, start=1):
            scores[scored.id] = scores.get(scored.id, 0.0) + 1.0 / (k + rank)
    merged = [ScoredId(id=item_id, score=score) for item_id, score in scores.items()]
    merged.sort(key=lambda s: (-s.score, s.id))
    return merged


class HybridRetriever:
    def __init__(
        self,
        keyword: KeywordRetriever,
        vector: VectorRetriever,
        metadata: MetadataStore,
    ) -> None:
        self._keyword = keyword
        self._vector = vector
        self._metadata = metadata

    async def search(
        self, query: str, filters: SearchFilters | None, top_n: int = 50
    ) -> list[RetrievalCandidate]:
        keyword_results = await self._keyword.search(query, filters, top_n)
        vector_results = await self._vector.search(query, filters, top_n)
        merged = rrf_merge([keyword_results, vector_results])
        chunks = await self._metadata.get_chunks([s.id for s in merged])
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        candidates = [
            RetrievalCandidate(chunk=chunks_by_id[s.id], rrf_score=s.score)
            for s in merged
            if s.id in chunks_by_id
        ]
        return _apply_temporal_filter(candidates, filters)


def _apply_temporal_filter(
    candidates: list[RetrievalCandidate], filters: SearchFilters | None
) -> list[RetrievalCandidate]:
    """Enforce temporal validity on hydrated chunks. This is the single
    authoritative check: vector payloads can go stale when chunks are expired
    in place (e.g. superseded decisions)."""
    if filters is not None and filters.include_expired:
        return candidates
    at = (filters.valid_at if filters else None) or utcnow()
    return [
        c
        for c in candidates
        if (c.chunk.valid_to is None or c.chunk.valid_to >= at)
        and (c.chunk.valid_from is None or c.chunk.valid_from <= at)
    ]


class SearchPipeline:
    """Tool-facing composition: hybrid retrieval, optional rerank, shaping."""

    def __init__(
        self,
        hybrid: HybridRetriever,
        rerank: RerankService,
        metadata: MetadataStore,
        settings: Settings,
    ) -> None:
        self._hybrid = hybrid
        self._rerank = rerank
        self._metadata = metadata
        self._settings = settings

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchOutput:
        use_rerank = self._settings.reranker.enabled_default if rerank is None else rerank
        fetch_n = max(50, top_k * 5)
        candidates = await self._hybrid.search(query, filters, fetch_n)
        if use_rerank:
            pool = candidates[: self._settings.reranker.max_candidates]
            selected = await self._rerank.rerank(query, pool, top_k)
        else:
            selected = candidates[:top_k]
        results = [await self._shape(candidate) for candidate in selected]
        return SearchOutput(results=results, reranked=use_rerank)

    async def _shape(self, candidate: RetrievalCandidate) -> SearchResult:
        chunk = candidate.chunk
        document = await self._metadata.get_document(chunk.document_id)
        return shape_result(candidate, document)


def shape_result(candidate: RetrievalCandidate, document: DocumentRecord | None) -> SearchResult:
    chunk: ChunkRecord = candidate.chunk
    doc_title = document.title if document else ""
    if not chunk.heading_path:
        title = doc_title
    elif not doc_title or chunk.heading_path.startswith(doc_title):
        title = chunk.heading_path
    else:
        title = f"{doc_title} > {chunk.heading_path}"
    source_path = chunk.path or (document.source_uri if document else "")
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "heading_path": chunk.heading_path,
            "ordinal": chunk.ordinal,
            "tags": chunk.tags,
            "rrf_score": candidate.rrf_score,
        }
    )
    score = candidate.rerank_score if candidate.rerank_score is not None else candidate.rrf_score
    return SearchResult(
        id=chunk.id,
        kind=chunk.kind,
        score=score,
        source_id=chunk.document_id,
        source_path=source_path,
        title=title,
        excerpt=make_excerpt(chunk.text),
        metadata=metadata,
    )
