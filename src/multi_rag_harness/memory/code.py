"""Code memory: source-level retrieval.

Code enters the index through ``rag_ingest_path`` (``kind="code"`` with
language detection); this service only scopes hybrid search to code chunks.
Symbol/call-graph extraction is a later target per the specification.
"""

from __future__ import annotations

from multi_rag_harness.retrieval.hybrid import SearchPipeline
from multi_rag_harness.retrieval.results import SearchOutput
from multi_rag_harness.storage.interfaces import SearchFilters


class CodeSearchService:
    def __init__(self, search: SearchPipeline) -> None:
        self._search = search

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchOutput:
        filters = (filters or SearchFilters()).model_copy(update={"kinds": ["code"]})
        return await self._search.search(query, filters, top_k, rerank)
