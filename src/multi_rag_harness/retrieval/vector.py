"""Vector retrieval services."""

from __future__ import annotations

from collections.abc import Sequence

from multi_rag_harness.models.embedding import EmbeddingModel
from multi_rag_harness.storage.interfaces import ScoredId, SearchFilters, VectorIndex


class VectorRetriever:
    """Embeds the query and searches the vector index."""

    def __init__(self, vector_index: VectorIndex, embedder: EmbeddingModel) -> None:
        self._index = vector_index
        self._embedder = embedder

    async def search(
        self,
        query: str,
        filters: SearchFilters | None,
        limit: int,
        item_kinds: Sequence[str] | None = None,
    ) -> list[ScoredId]:
        vectors = await self._embedder.embed_queries([query])
        return await self._index.search(vectors[0], filters, limit, item_kinds=item_kinds)
