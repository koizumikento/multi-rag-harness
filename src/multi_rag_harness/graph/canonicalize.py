"""Entity canonicalization: name normalization and alias-based merging."""

from __future__ import annotations

import unicodedata

from multi_rag_harness.graph.models import ExtractedEntity
from multi_rag_harness.storage.interfaces import (
    AliasRecord,
    EntityNode,
    GraphStore,
    MetadataStore,
)


def normalize_name(name: str) -> str:
    """NFKC-normalize, casefold, and collapse whitespace."""
    normalized = unicodedata.normalize("NFKC", name)
    return " ".join(normalized.casefold().split())


class EntityCanonicalizer:
    def __init__(self, metadata: MetadataStore, graph: GraphStore) -> None:
        self._metadata = metadata
        self._graph = graph

    async def resolve(
        self, extracted: ExtractedEntity, source_id: str | None = None
    ) -> tuple[str, bool]:
        """Resolve an extracted entity to a canonical entity id.

        Returns ``(entity_id, created)``. On an alias hit the existing entity
        is merged (new aliases added, empty fields filled); on a miss a new
        entity is created with a self-alias.
        """
        candidate_names = [extracted.name, *extracted.aliases]
        normalized_candidates = []
        for name in candidate_names:
            normalized = normalize_name(name)
            if normalized and normalized not in normalized_candidates:
                normalized_candidates.append(normalized)
        if not normalized_candidates:
            raise ValueError("extracted entity has no usable name")

        for normalized in normalized_candidates:
            entity_id = await self._metadata.find_entity_id_by_normalized_alias(normalized)
            if entity_id is not None:
                await self._merge(entity_id, extracted, candidate_names, source_id)
                return entity_id, False

        entity = EntityNode(
            canonical_name=extracted.name,
            entity_type=extracted.type,
            description=extracted.description,
        )
        await self._graph.upsert_entity(entity)
        await self._insert_aliases(entity.id, candidate_names, extracted.confidence, source_id)
        return entity.id, True

    async def _merge(
        self,
        entity_id: str,
        extracted: ExtractedEntity,
        candidate_names: list[str],
        source_id: str | None,
    ) -> None:
        entity = await self._graph.get_entity(entity_id)
        if entity is not None:
            changed = False
            if not entity.description and extracted.description:
                entity.description = extracted.description
                changed = True
            if not entity.entity_type and extracted.type:
                entity.entity_type = extracted.type
                changed = True
            if changed:
                await self._graph.upsert_entity(entity)
        await self._insert_aliases(entity_id, candidate_names, extracted.confidence, source_id)

    async def _insert_aliases(
        self,
        entity_id: str,
        names: list[str],
        confidence: float,
        source_id: str | None,
    ) -> None:
        aliases = []
        seen: set[str] = set()
        for name in names:
            normalized = normalize_name(name)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            aliases.append(
                AliasRecord(
                    entity_id=entity_id,
                    alias=name,
                    normalized_alias=normalized,
                    source_id=source_id,
                    confidence=confidence,
                )
            )
        await self._metadata.insert_aliases(aliases)
