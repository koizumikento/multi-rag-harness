"""Integration tests for graph community storage helpers."""

from multi_rag_harness.graph.communities import CommunityService
from multi_rag_harness.storage.interfaces import DocumentRecord, EntityNode
from tests.conftest import Harness


async def test_community_service_stores_members_and_provenance(harness: Harness) -> None:
    entity = EntityNode(canonical_name="Kuzu", entity_type="technology")
    await harness.storage.graph.upsert_entity(entity)
    document = DocumentRecord(
        source_uri="memory://community-source",
        title="Community Source",
        content_hash="community-source",
    )
    await harness.storage.metadata.upsert_document(document)

    service = CommunityService(harness.storage.graph, harness.storage.metadata)
    community = await service.store_community(
        "Graph Stores",
        "Graph storage related entities.",
        1,
        [entity.id],
        provenance_source_id=document.id,
    )

    assert await service.get_for_entity(entity.id) == [community]
    assert await service.list_communities(limit=10) == [community]
    sources = await harness.storage.metadata.get_provenance_for_item(community.id)
    assert sources[0].source_id == document.id
