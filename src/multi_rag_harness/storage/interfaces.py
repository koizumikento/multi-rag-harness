"""Storage service interfaces and shared record models.

Backend-neutral contracts. Nothing in this module imports concrete backends.
All identifiers are uuid4 strings. All timestamps are timezone-aware UTC
datetimes; adapters own their serialized representation.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    source_type: str = "file"  # file | memory | graph
    source_uri: str
    title: str
    content_hash: str
    scope: str = "default"
    kind: str = "doc"
    repo: str | None = None
    language: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ChunkRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    document_id: str
    ordinal: int
    heading_path: str = ""
    text: str
    token_count: int
    # Denormalized filter columns copied from the parent document.
    scope: str = "default"
    kind: str = "doc"
    repo: str | None = None
    path: str | None = None
    language: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_type: str = "file"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ScoredId(BaseModel):
    id: str
    score: float


class VectorPoint(BaseModel):
    id: str
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class ProvenanceRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    item_type: str  # entity | relation | claim | alias | community
    item_id: str
    source_id: str
    chunk_id: str | None = None
    evidence_text: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    extraction_run_id: str | None = None


class ExtractionRunRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    source_id: str
    chunk_id: str
    codex_thread_id: str | None = None
    prompt_version: str
    status: str = "pending"  # pending | running | completed | failed
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class TraceRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    task: str
    outcome: str
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
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class DecisionRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str
    status: str = "accepted"  # proposed | accepted | rejected | superseded
    context: str | None = None
    decision: str
    rationale: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    consequences: str | None = None
    source_links: list[str] = Field(default_factory=list)
    related_entities: list[str] = Field(default_factory=list)
    supersedes: str | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class FailureRecord(BaseModel):
    id: str = Field(default_factory=new_id)
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
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class ToolRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    server: str
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_shape: str | None = None
    approval_policy: str | None = None
    rate_limits: str | None = None
    examples: list[str] = Field(default_factory=list)
    known_failure_modes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ClaimRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    subject_entity_id: str
    predicate: str
    object_text: str
    object_entity_id: str | None = None
    modality: str = "fact"  # fact | hypothesis | decision | requirement | constraint
    confidence: float = 1.0
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class AliasRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    entity_id: str
    alias: str
    normalized_alias: str
    source_id: str | None = None
    confidence: float = 1.0


class EntityNode(BaseModel):
    id: str = Field(default_factory=new_id)
    canonical_name: str
    entity_type: str = ""
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class RelationEdge(BaseModel):
    id: str = Field(default_factory=new_id)
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str = ""
    confidence: float = 1.0
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class CommunityNode(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str
    summary: str = ""
    level: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class SearchFilters(BaseModel):
    scopes: list[str] | None = None
    kinds: list[str] | None = None
    repo: str | None = None
    path_prefix: str | None = None
    language: str | None = None
    tags: list[str] | None = None
    source_type: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    valid_at: datetime | None = None  # temporal point-in-time; None means "now"
    include_expired: bool = False
    confidence_min: float | None = None  # graph items only


@runtime_checkable
class MetadataStore(Protocol):
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...

    # documents
    async def upsert_document(self, doc: DocumentRecord) -> None: ...
    async def get_document(self, document_id: str) -> DocumentRecord | None: ...
    async def find_document_by_uri(self, source_uri: str, scope: str) -> DocumentRecord | None: ...
    async def delete_document(self, document_id: str) -> None: ...

    # chunks
    async def insert_chunks(self, chunks: Sequence[ChunkRecord]) -> None: ...
    async def get_chunk(self, chunk_id: str) -> ChunkRecord | None: ...
    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[ChunkRecord]: ...
    async def get_chunk_window(self, chunk_id: str, around: int) -> list[ChunkRecord]: ...
    async def get_chunks_for_document(
        self, document_id: str, limit: int | None = None
    ) -> list[ChunkRecord]: ...
    async def delete_chunks_for_document(self, document_id: str) -> list[str]:
        """Delete all chunks of a document. Returns their embedding ids."""
        ...

    async def expire_chunks_for_document(self, document_id: str, valid_to: datetime) -> None: ...

    # provenance and extraction runs
    async def insert_provenance(self, records: Sequence[ProvenanceRecord]) -> None: ...
    async def get_provenance_for_item(self, item_id: str) -> list[ProvenanceRecord]: ...
    async def get_entity_ids_for_chunks(self, chunk_ids: Sequence[str]) -> list[str]: ...
    async def create_extraction_runs(self, runs: Sequence[ExtractionRunRecord]) -> None: ...
    async def get_extraction_run(self, run_id: str) -> ExtractionRunRecord | None: ...
    async def claim_pending_extraction_runs(self, limit: int) -> list[ExtractionRunRecord]:
        """Atomically move up to ``limit`` pending runs to running and return them."""
        ...

    async def update_extraction_run(
        self,
        run_id: str,
        *,
        status: str,
        codex_thread_id: str | None = None,
        error: str | None = None,
    ) -> None: ...

    # memory records
    async def insert_trace(self, record: TraceRecord) -> None: ...
    async def get_trace(self, trace_id: str) -> TraceRecord | None: ...
    async def insert_decision(self, record: DecisionRecord) -> None: ...
    async def get_decision(self, decision_id: str) -> DecisionRecord | None: ...
    async def mark_decision_superseded(self, old_id: str, new_id: str) -> None: ...
    async def insert_failure(self, record: FailureRecord) -> None: ...
    async def get_failure(self, failure_id: str) -> FailureRecord | None: ...
    async def upsert_tool_record(self, record: ToolRecord) -> ToolRecord:
        """Insert or update on (server, name). Returns the stored record."""
        ...

    async def get_tool_record(self, tool_id: str) -> ToolRecord | None: ...

    # graph relational data
    async def insert_claims(self, claims: Sequence[ClaimRecord]) -> None: ...
    async def get_claim(self, claim_id: str) -> ClaimRecord | None: ...
    async def get_claims(self, claim_ids: Sequence[str]) -> list[ClaimRecord]: ...
    async def get_claims_for_entity(
        self, entity_id: str, valid_at: datetime | None = None, limit: int = 20
    ) -> list[ClaimRecord]: ...
    async def insert_aliases(self, aliases: Sequence[AliasRecord]) -> None: ...
    async def get_aliases_for_entity(self, entity_id: str) -> list[AliasRecord]: ...
    async def find_entity_id_by_normalized_alias(self, normalized: str) -> str | None: ...


@runtime_checkable
class KeywordIndex(Protocol):
    """Full-text keyword index. Queries are raw natural-language text;
    adapters own backend-specific escaping."""

    async def index_chunks(self, chunks: Sequence[ChunkRecord], title: str) -> None: ...
    async def remove_chunks(self, chunk_ids: Sequence[str]) -> None: ...
    async def search_chunks(
        self, query: str, filters: SearchFilters | None, limit: int
    ) -> list[ScoredId]: ...
    async def index_graph_item(self, item_id: str, item_type: str, text: str) -> None: ...
    async def remove_graph_item(self, item_id: str) -> None: ...
    async def search_graph_items(
        self, query: str, item_type: str, limit: int
    ) -> list[ScoredId]: ...


@runtime_checkable
class VectorIndex(Protocol):
    async def initialize(self, dimension: int) -> None: ...
    async def close(self) -> None: ...
    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...
    async def delete(self, point_ids: Sequence[str]) -> None: ...
    async def search(
        self,
        vector: Sequence[float],
        filters: SearchFilters | None,
        limit: int,
        item_kinds: Sequence[str] | None = None,
    ) -> list[ScoredId]:
        """Returned ``ScoredId.id`` is the payload ``item_id``, never the point id."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def upsert_entity(self, entity: EntityNode) -> None: ...
    async def get_entity(self, entity_id: str) -> EntityNode | None: ...
    async def get_entities(self, entity_ids: Sequence[str]) -> list[EntityNode]: ...
    async def upsert_relation(self, relation: RelationEdge) -> None: ...
    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: Sequence[str] | None = None,
        valid_at: datetime | None = None,
    ) -> list[tuple[RelationEdge, EntityNode]]: ...
    async def upsert_community(
        self, community: CommunityNode, member_entity_ids: Sequence[str]
    ) -> None: ...
    async def get_communities_for_entity(self, entity_id: str) -> list[CommunityNode]: ...
    async def list_communities(self, limit: int = 50) -> list[CommunityNode]: ...
