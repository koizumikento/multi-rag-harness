"""Failure/error memory: error signatures and resolution history."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from multi_rag_harness.ingestion.pipeline import IngestionPipeline
from multi_rag_harness.memory import StoredMemoryRef, render_list_section, render_section
from multi_rag_harness.retrieval.hybrid import SearchPipeline
from multi_rag_harness.retrieval.results import SearchOutput
from multi_rag_harness.storage.interfaces import FailureRecord, MetadataStore, SearchFilters


class FailurePayload(BaseModel):
    error_text: str
    error_category: str | None = None
    command: str | None = None
    environment: str | None = None
    suspected_cause: str | None = None
    confirmed_cause: str | None = None
    fix_applied: str | None = None
    verification: str | None = None
    related_traces: list[str] = Field(default_factory=list)
    related_code_paths: list[str] = Field(default_factory=list)
    scope: str = "default"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render_text(self) -> str:
        return (
            "# Failure\n\n"
            + render_section("Error", self.error_text)
            + render_section("Category", self.error_category)
            + render_section("Command", self.command)
            + render_section("Environment", self.environment)
            + render_section("Suspected Cause", self.suspected_cause)
            + render_section("Confirmed Cause", self.confirmed_cause)
            + render_section("Fix Applied", self.fix_applied)
            + render_section("Verification", self.verification)
            + render_list_section("Related Code Paths", self.related_code_paths)
        )


class FailureMemoryService:
    def __init__(
        self,
        metadata: MetadataStore,
        pipeline: IngestionPipeline,
        search: SearchPipeline,
    ) -> None:
        self._metadata = metadata
        self._pipeline = pipeline
        self._search = search

    async def store(self, payload: FailurePayload) -> StoredMemoryRef:
        record = FailureRecord(
            error_text=payload.error_text,
            error_category=payload.error_category,
            command=payload.command,
            environment=payload.environment,
            suspected_cause=payload.suspected_cause,
            confirmed_cause=payload.confirmed_cause,
            fix_applied=payload.fix_applied,
            verification=payload.verification,
            related_traces=payload.related_traces,
            related_code_paths=payload.related_code_paths,
            metadata=payload.metadata,
        )
        await self._metadata.insert_failure(record)
        document_id = await self._pipeline.ingest_memory_record(
            kind="error",
            record_id=record.id,
            title=f"Failure: {payload.error_text[:100]}",
            text=payload.render_text(),
            scope=payload.scope,
            tags=payload.tags,
        )
        return StoredMemoryRef(record_id=record.id, document_id=document_id)

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchOutput:
        filters = (filters or SearchFilters()).model_copy(update={"kinds": ["error"]})
        return await self._search.search(query, filters, top_k, rerank)
