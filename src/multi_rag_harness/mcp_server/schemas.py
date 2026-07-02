"""MCP tool input and output schemas.

Response models stay compact (summaries, truncated excerpts) so tool output
fits agent context; ``rag_get_source`` / ``graph_get_sources`` provide the
full-context expansion path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from multi_rag_harness.graph.models import GraphNeighborhood, GraphSourceRef
from multi_rag_harness.ingestion.pipeline import IngestReport
from multi_rag_harness.retrieval.results import SearchResult
from multi_rag_harness.storage.interfaces import SearchFilters


class FiltersInput(BaseModel):
    """Agent-facing metadata filters, shared by every search tool."""

    repo: str | None = None
    path_prefix: str | None = None
    language: str | None = None
    tags: list[str] | None = None
    source_type: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    valid_at: datetime | None = None
    include_expired: bool = False
    confidence_min: float | None = None

    def to_search_filters(
        self,
        scopes: list[str] | None = None,
        kinds: list[str] | None = None,
    ) -> SearchFilters:
        return SearchFilters(
            scopes=scopes,
            kinds=kinds,
            repo=self.repo,
            path_prefix=self.path_prefix,
            language=self.language,
            tags=self.tags,
            source_type=self.source_type,
            created_after=self.created_after,
            created_before=self.created_before,
            valid_at=self.valid_at,
            include_expired=self.include_expired,
            confidence_min=self.confidence_min,
        )


def resolve_filters(
    filters: FiltersInput | None,
    scopes: list[str] | None = None,
    kinds: list[str] | None = None,
) -> SearchFilters:
    return (filters or FiltersInput()).to_search_filters(scopes, kinds)


class IngestOptions(BaseModel):
    kind: str | None = None  # override detected kind (e.g. force "doc")
    extract: bool | None = None  # queue graph extraction runs for new chunks


class IngestResponse(IngestReport):
    pass


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    reranked: bool


class ChunkContext(BaseModel):
    chunk_id: str
    ordinal: int
    heading_path: str
    text: str
    is_target: bool


class GetSourceResponse(BaseModel):
    source_id: str
    source_path: str
    title: str
    kind: str
    chunks: list[ChunkContext]
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntitySummary(BaseModel):
    id: str
    canonical_name: str
    entity_type: str = ""
    description: str = ""


class RelationSummary(BaseModel):
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str = ""
    confidence: float = 1.0


class ClaimSummary(BaseModel):
    id: str
    subject_entity_id: str
    predicate: str
    object_text: str
    modality: str
    confidence: float


class GraphExpandResponse(BaseModel):
    root_entity_id: str
    entities: list[EntitySummary]
    relations: list[RelationSummary]
    claims: list[ClaimSummary]

    @classmethod
    def from_neighborhood(cls, neighborhood: GraphNeighborhood) -> GraphExpandResponse:
        return cls(
            root_entity_id=neighborhood.root_entity_id,
            entities=[
                EntitySummary(
                    id=e.id,
                    canonical_name=e.canonical_name,
                    entity_type=e.entity_type,
                    description=e.description,
                )
                for e in neighborhood.entities
            ],
            relations=[
                RelationSummary(
                    id=r.id,
                    source_entity_id=r.source_entity_id,
                    target_entity_id=r.target_entity_id,
                    relation_type=r.relation_type,
                    description=r.description,
                    confidence=r.confidence,
                )
                for r in neighborhood.relations
            ],
            claims=[
                ClaimSummary(
                    id=c.id,
                    subject_entity_id=c.subject_entity_id,
                    predicate=c.predicate,
                    object_text=c.object_text,
                    modality=c.modality,
                    confidence=c.confidence,
                )
                for c in neighborhood.claims
            ],
        )


class GraphSourcesResponse(BaseModel):
    graph_item_id: str
    sources: list[GraphSourceRef]


class StoreResponse(BaseModel):
    record_id: str
    document_id: str
