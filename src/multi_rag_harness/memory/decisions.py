"""Decision memory: durable technical choices and rationale."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from multi_rag_harness.ingestion.pipeline import IngestionPipeline
from multi_rag_harness.memory import StoredMemoryRef, render_list_section, render_section
from multi_rag_harness.retrieval.hybrid import SearchPipeline
from multi_rag_harness.retrieval.results import SearchOutput
from multi_rag_harness.storage.interfaces import (
    DecisionRecord,
    MetadataStore,
    SearchFilters,
    utcnow,
)


class DecisionPayload(BaseModel):
    title: str
    decision: str
    status: Literal["proposed", "accepted", "rejected", "superseded"] = "accepted"
    context: str | None = None
    rationale: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    consequences: str | None = None
    source_links: list[str] = Field(default_factory=list)
    related_entities: list[str] = Field(default_factory=list)
    supersedes: str | None = None
    scope: str = "default"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render_text(self) -> str:
        return (
            f"# Decision: {self.title}\n\n"
            + render_section("Status", self.status)
            + render_section("Context", self.context)
            + render_section("Decision", self.decision)
            + render_section("Rationale", self.rationale)
            + render_list_section("Alternatives Considered", self.alternatives)
            + render_section("Consequences", self.consequences)
            + render_list_section("Sources", self.source_links)
        )


class DecisionMemoryService:
    def __init__(
        self,
        metadata: MetadataStore,
        pipeline: IngestionPipeline,
        search: SearchPipeline,
    ) -> None:
        self._metadata = metadata
        self._pipeline = pipeline
        self._search = search

    async def store(self, payload: DecisionPayload) -> StoredMemoryRef:
        record = DecisionRecord(
            title=payload.title,
            status=payload.status,
            context=payload.context,
            decision=payload.decision,
            rationale=payload.rationale,
            alternatives=payload.alternatives,
            consequences=payload.consequences,
            source_links=payload.source_links,
            related_entities=payload.related_entities,
            supersedes=payload.supersedes,
            metadata=payload.metadata,
        )
        await self._metadata.insert_decision(record)
        if payload.supersedes:
            await self._supersede(payload.supersedes, record.id, payload.scope)
        document_id = await self._pipeline.ingest_memory_record(
            kind="decision",
            record_id=record.id,
            title=f"Decision: {payload.title[:100]}",
            text=payload.render_text(),
            scope=payload.scope,
            tags=payload.tags,
        )
        return StoredMemoryRef(record_id=record.id, document_id=document_id)

    async def _supersede(self, old_id: str, new_id: str, scope: str) -> None:
        """Mark the old decision superseded and expire its search chunks so
        temporal filtering hides it from default searches (Temporal RAG)."""
        await self._metadata.mark_decision_superseded(old_id, new_id)
        old_document = await self._metadata.find_document_by_uri(
            f"memory://decision/{old_id}", scope
        )
        if old_document is not None:
            await self._metadata.expire_chunks_for_document(old_document.id, utcnow())

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchOutput:
        filters = (filters or SearchFilters()).model_copy(update={"kinds": ["decision"]})
        return await self._search.search(query, filters, top_k, rerank)
