"""Integration tests for the Kuzu graph store."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from multi_rag_harness.storage.interfaces import CommunityNode, EntityNode, RelationEdge
from multi_rag_harness.storage.kuzu import KuzuGraphStore


@pytest.fixture
async def graph(tmp_path: Path):
    store = KuzuGraphStore(tmp_path / "kuzu")
    await store.initialize()
    yield store
    await store.close()


async def test_entity_upsert_and_reupsert(graph: KuzuGraphStore) -> None:
    entity = EntityNode(canonical_name="Qdrant", entity_type="technology", description="vec db")
    await graph.upsert_entity(entity)
    loaded = await graph.get_entity(entity.id)
    assert loaded is not None
    assert loaded.canonical_name == "Qdrant"
    assert loaded.entity_type == "technology"

    entity.description = "vector database"
    await graph.upsert_entity(entity)
    reloaded = await graph.get_entity(entity.id)
    assert reloaded is not None and reloaded.description == "vector database"

    assert await graph.get_entity("missing") is None


async def test_get_entities_bulk(graph: KuzuGraphStore) -> None:
    entities = [EntityNode(canonical_name=f"E{i}") for i in range(3)]
    for entity in entities:
        await graph.upsert_entity(entity)
    loaded = await graph.get_entities([entities[2].id, entities[0].id])
    assert [e.canonical_name for e in loaded] == ["E2", "E0"]


async def test_relations_and_neighbors(graph: KuzuGraphStore) -> None:
    a = EntityNode(canonical_name="A")
    b = EntityNode(canonical_name="B")
    c = EntityNode(canonical_name="C")
    for entity in (a, b, c):
        await graph.upsert_entity(entity)

    ab = RelationEdge(
        source_entity_id=a.id, target_entity_id=b.id, relation_type="uses", confidence=0.9
    )
    ca = RelationEdge(
        source_entity_id=c.id, target_entity_id=a.id, relation_type="depends_on", confidence=0.8
    )
    await graph.upsert_relation(ab)
    await graph.upsert_relation(ca)

    neighbors = await graph.get_neighbors(a.id)
    by_name = {entity.canonical_name: rel for rel, entity in neighbors}
    assert set(by_name) == {"B", "C"}
    assert by_name["B"].relation_type == "uses"
    assert by_name["B"].source_entity_id == a.id
    assert by_name["B"].target_entity_id == b.id
    assert by_name["C"].source_entity_id == c.id  # incoming edge keeps its direction

    filtered = await graph.get_neighbors(a.id, relation_types=["uses"])
    assert [entity.canonical_name for _, entity in filtered] == ["B"]


async def test_relation_reupsert_updates(graph: KuzuGraphStore) -> None:
    a = EntityNode(canonical_name="A")
    b = EntityNode(canonical_name="B")
    for entity in (a, b):
        await graph.upsert_entity(entity)
    rel = RelationEdge(source_entity_id=a.id, target_entity_id=b.id, relation_type="uses")
    await graph.upsert_relation(rel)
    rel.confidence = 0.5
    await graph.upsert_relation(rel)

    neighbors = await graph.get_neighbors(a.id)
    assert len(neighbors) == 1
    assert neighbors[0][0].confidence == 0.5


async def test_temporal_relation_filter(graph: KuzuGraphStore) -> None:
    a = EntityNode(canonical_name="A")
    b = EntityNode(canonical_name="B")
    for entity in (a, b):
        await graph.upsert_entity(entity)
    expired = RelationEdge(
        source_entity_id=a.id,
        target_entity_id=b.id,
        relation_type="used",
        valid_to=datetime(2020, 1, 1, tzinfo=UTC),
    )
    await graph.upsert_relation(expired)

    assert await graph.get_neighbors(a.id) == []
    past = await graph.get_neighbors(a.id, valid_at=datetime(2019, 1, 1, tzinfo=UTC))
    assert len(past) == 1


async def test_communities(graph: KuzuGraphStore) -> None:
    a = EntityNode(canonical_name="A")
    b = EntityNode(canonical_name="B")
    for entity in (a, b):
        await graph.upsert_entity(entity)
    community = CommunityNode(title="Storage", summary="storage backends", level=0)
    await graph.upsert_community(community, [a.id, b.id])

    for_a = await graph.get_communities_for_entity(a.id)
    assert [c.title for c in for_a] == ["Storage"]

    all_communities = await graph.list_communities()
    assert [c.title for c in all_communities] == ["Storage"]

    # Re-upsert is idempotent for membership.
    await graph.upsert_community(community, [a.id])
    assert len(await graph.get_communities_for_entity(a.id)) == 1
