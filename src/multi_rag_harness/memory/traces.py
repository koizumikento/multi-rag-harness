"""Trace memory: durable agent/task execution history."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from multi_rag_harness.ingestion.pipeline import IngestionPipeline
from multi_rag_harness.memory import StoredMemoryRef, render_list_section, render_section
from multi_rag_harness.retrieval.hybrid import SearchPipeline
from multi_rag_harness.retrieval.results import SearchOutput
from multi_rag_harness.storage.interfaces import MetadataStore, SearchFilters, TraceRecord


class TracePayload(BaseModel):
    task: str
    outcome: str
    prompt_summary: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    final_response: str | None = None
    human_feedback: str | None = None
    linked_decisions: list[str] = Field(default_factory=list)
    linked_entities: list[str] = Field(default_factory=list)
    scope: str = "default"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render_text(self) -> str:
        return (
            f"# Trace: {self.task}\n\n"
            + render_section("Task", self.task)
            + render_section("Prompt Summary", self.prompt_summary)
            + render_section("Outcome", self.outcome)
            + render_list_section("Tools Used", self.tools_used)
            + render_list_section("Commands", self.commands)
            + render_list_section("Files Read", self.files_read)
            + render_list_section("Files Changed", self.files_changed)
            + render_list_section("Errors", self.errors)
            + render_list_section("Tests", self.tests)
            + render_section("Final Response", self.final_response)
            + render_section("Human Feedback", self.human_feedback)
        )


class TraceMemoryService:
    def __init__(
        self,
        metadata: MetadataStore,
        pipeline: IngestionPipeline,
        search: SearchPipeline,
    ) -> None:
        self._metadata = metadata
        self._pipeline = pipeline
        self._search = search

    async def store(self, payload: TracePayload) -> StoredMemoryRef:
        record_metadata = dict(payload.metadata)
        if payload.prompt_summary:
            record_metadata["prompt_summary"] = payload.prompt_summary
        record = TraceRecord(
            task=payload.task,
            outcome=payload.outcome,
            tools_used=payload.tools_used,
            commands=payload.commands,
            files_read=payload.files_read,
            files_changed=payload.files_changed,
            errors=payload.errors,
            tests=payload.tests,
            final_response=payload.final_response,
            human_feedback=payload.human_feedback,
            linked_decisions=payload.linked_decisions,
            linked_entities=payload.linked_entities,
            metadata=record_metadata,
        )
        await self._metadata.insert_trace(record)
        document_id = await self._pipeline.ingest_memory_record(
            kind="trace",
            record_id=record.id,
            title=f"Trace: {payload.task[:100]}",
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
        filters = (filters or SearchFilters()).model_copy(update={"kinds": ["trace"]})
        return await self._search.search(query, filters, top_k, rerank)
