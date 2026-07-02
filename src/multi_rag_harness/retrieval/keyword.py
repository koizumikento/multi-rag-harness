"""Keyword retrieval services."""

from __future__ import annotations

from multi_rag_harness.storage.interfaces import KeywordIndex, ScoredId, SearchFilters


class KeywordRetriever:
    """Thin service over the backend keyword index; queries are raw text."""

    def __init__(self, keyword_index: KeywordIndex) -> None:
        self._index = keyword_index

    async def search(self, query: str, filters: SearchFilters | None, limit: int) -> list[ScoredId]:
        return await self._index.search_chunks(query, filters, limit)
