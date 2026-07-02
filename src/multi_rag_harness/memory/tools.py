"""Tool memory: MCP tool descriptions for tool selection support.

Stored via the Python API (or CLI later); the MCP tool surface only exposes
``tool_search`` per the specification.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from multi_rag_harness.ingestion.pipeline import IngestionPipeline
from multi_rag_harness.memory import StoredMemoryRef, render_list_section, render_section
from multi_rag_harness.retrieval.hybrid import SearchPipeline
from multi_rag_harness.retrieval.results import SearchOutput
from multi_rag_harness.storage.interfaces import MetadataStore, SearchFilters, ToolRecord


class ToolRecordPayload(BaseModel):
    server: str
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_shape: str | None = None
    approval_policy: str | None = None
    rate_limits: str | None = None
    examples: list[str] = Field(default_factory=list)
    known_failure_modes: list[str] = Field(default_factory=list)
    scope: str = "default"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render_text(self) -> str:
        schema_block = (
            f"```json\n{json.dumps(self.input_schema, indent=2, ensure_ascii=False)}\n```"
            if self.input_schema
            else ""
        )
        return (
            f"# Tool: {self.server}/{self.name}\n\n"
            + render_section("Description", self.description)
            + render_section("Input Schema", schema_block)
            + render_section("Output Shape", self.output_shape)
            + render_section("Approval Policy", self.approval_policy)
            + render_section("Rate Limits", self.rate_limits)
            + render_list_section("Examples", self.examples)
            + render_list_section("Known Failure Modes", self.known_failure_modes)
        )


class ToolMemoryService:
    def __init__(
        self,
        metadata: MetadataStore,
        pipeline: IngestionPipeline,
        search: SearchPipeline,
    ) -> None:
        self._metadata = metadata
        self._pipeline = pipeline
        self._search = search

    async def store(self, payload: ToolRecordPayload) -> StoredMemoryRef:
        record = ToolRecord(
            server=payload.server,
            name=payload.name,
            description=payload.description,
            input_schema=payload.input_schema,
            output_shape=payload.output_shape,
            approval_policy=payload.approval_policy,
            rate_limits=payload.rate_limits,
            examples=payload.examples,
            known_failure_modes=payload.known_failure_modes,
            metadata=payload.metadata,
        )
        stored = await self._metadata.upsert_tool_record(record)
        document_id = await self._pipeline.ingest_memory_record(
            kind="tool",
            record_id=stored.id,
            title=f"Tool: {payload.server}/{payload.name}",
            text=payload.render_text(),
            scope=payload.scope,
            tags=payload.tags,
        )
        return StoredMemoryRef(record_id=stored.id, document_id=document_id)

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchOutput:
        filters = (filters or SearchFilters()).model_copy(update={"kinds": ["tool"]})
        return await self._search.search(query, filters, top_k, rerank)
