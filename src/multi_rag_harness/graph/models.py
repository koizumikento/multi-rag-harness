"""Graph domain models.

The ``Extracted*`` models mirror the Codex extraction JSON schema from the
specification exactly; validation of raw extraction output happens in
``graph.extraction``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from multi_rag_harness.storage.interfaces import ClaimRecord, EntityNode, RelationEdge

Modality = Literal["fact", "hypothesis", "decision", "requirement", "constraint"]


class ExtractedEntity(BaseModel):
    name: str
    type: str = ""
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class ExtractedRelation(BaseModel):
    source: str
    target: str
    type: str
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    valid_from: str | None = None
    valid_to: str | None = None


class ExtractedClaim(BaseModel):
    subject: str
    predicate: str
    object: str
    modality: Modality = "fact"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    valid_from: str | None = None
    valid_to: str | None = None


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)


class ValidatedExtraction(BaseModel):
    result: ExtractionResult
    warnings: list[str] = Field(default_factory=list)
    rejected_count: int = 0


class GraphNeighborhood(BaseModel):
    root_entity_id: str
    entities: list[EntityNode] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)
    claims: list[ClaimRecord] = Field(default_factory=list)


class GraphSourceRef(BaseModel):
    """Provenance-backed source pointer for a graph item."""

    source_id: str
    source_path: str
    chunk_id: str | None = None
    evidence_text: str | None = None
    excerpt: str = ""
