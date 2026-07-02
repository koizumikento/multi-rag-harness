"""Graph community storage and retrieval.

Community detection (e.g. Leiden over the entity graph) is a documented TODO;
this service only persists and retrieves externally provided communities and
their summaries.
"""

from __future__ import annotations

from collections.abc import Sequence

from multi_rag_harness.storage.interfaces import (
    CommunityNode,
    GraphStore,
    MetadataStore,
    ProvenanceRecord,
)


class CommunityService:
    def __init__(self, graph: GraphStore, metadata: MetadataStore) -> None:
        self._graph = graph
        self._metadata = metadata

    async def store_community(
        self,
        title: str,
        summary: str,
        level: int,
        member_entity_ids: Sequence[str],
        provenance_source_id: str | None = None,
    ) -> CommunityNode:
        community = CommunityNode(title=title, summary=summary, level=level)
        await self._graph.upsert_community(community, member_entity_ids)
        if provenance_source_id is not None:
            await self._metadata.insert_provenance(
                [
                    ProvenanceRecord(
                        item_type="community",
                        item_id=community.id,
                        source_id=provenance_source_id,
                    )
                ]
            )
        return community

    async def get_for_entity(self, entity_id: str) -> list[CommunityNode]:
        return await self._graph.get_communities_for_entity(entity_id)

    async def list_communities(self, limit: int = 50) -> list[CommunityNode]:
        return await self._graph.list_communities(limit)
