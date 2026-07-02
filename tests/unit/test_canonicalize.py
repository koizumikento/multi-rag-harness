"""Unit tests for entity canonicalization."""

from pathlib import Path

import pytest

from multi_rag_harness.graph.canonicalize import EntityCanonicalizer, normalize_name
from multi_rag_harness.graph.models import ExtractedEntity
from multi_rag_harness.storage.interfaces import EntityNode
from multi_rag_harness.storage.kuzu import KuzuGraphStore
from multi_rag_harness.storage.sqlite import SqliteStore


def test_normalize_name_casefold_and_whitespace() -> None:
    assert normalize_name("  Kuzu   DB ") == "kuzu db"


def test_normalize_name_nfkc_fullwidth() -> None:
    assert normalize_name("Ｑｄｒａｎｔ") == "qdrant"


def test_normalize_name_japanese() -> None:
    assert normalize_name("グラフ  データベース") == "グラフ データベース"


@pytest.fixture
async def stores(tmp_path: Path):
    metadata = SqliteStore(tmp_path / "meta.db")
    graph = KuzuGraphStore(tmp_path / "kuzu")
    await metadata.initialize()
    await graph.initialize()
    yield metadata, graph
    await metadata.close()
    await graph.close()


async def test_resolve_creates_then_merges(stores) -> None:
    metadata, graph = stores
    canonicalizer = EntityCanonicalizer(metadata, graph)

    extracted = ExtractedEntity(
        name="Kuzu",
        type="technology",
        aliases=["KuzuDB"],
        description="graph db",
        confidence=0.9,
        evidence="e",
    )
    entity_id, created = await canonicalizer.resolve(extracted, source_id="doc-1")
    assert created is True

    # Same name resolves to the same entity.
    again = ExtractedEntity(name="kuzu", confidence=0.8, evidence="e")
    same_id, created2 = await canonicalizer.resolve(again, source_id="doc-2")
    assert same_id == entity_id
    assert created2 is False

    # Alias also resolves.
    via_alias = ExtractedEntity(name="KuzuDB", confidence=0.8, evidence="e")
    alias_id, created3 = await canonicalizer.resolve(via_alias)
    assert alias_id == entity_id
    assert created3 is False


async def test_merge_fills_empty_fields_and_adds_aliases(stores) -> None:
    metadata, graph = stores
    canonicalizer = EntityCanonicalizer(metadata, graph)

    bare = EntityNode(canonical_name="Qdrant")
    await graph.upsert_entity(bare)
    await metadata.insert_aliases([])
    # Register the existing entity's self-alias manually.
    from multi_rag_harness.storage.interfaces import AliasRecord

    await metadata.insert_aliases(
        [AliasRecord(entity_id=bare.id, alias="Qdrant", normalized_alias="qdrant")]
    )

    extracted = ExtractedEntity(
        name="Qdrant",
        type="technology",
        aliases=["qdrant-client"],
        description="vector database",
        confidence=1.0,
        evidence="e",
    )
    entity_id, created = await canonicalizer.resolve(extracted)
    assert entity_id == bare.id
    assert created is False

    merged = await graph.get_entity(bare.id)
    assert merged is not None
    assert merged.description == "vector database"
    assert merged.entity_type == "technology"

    aliases = await metadata.get_aliases_for_entity(bare.id)
    assert {a.normalized_alias for a in aliases} == {"qdrant", "qdrant-client"}


async def test_resolve_requires_usable_name(stores) -> None:
    metadata, graph = stores
    canonicalizer = EntityCanonicalizer(metadata, graph)
    with pytest.raises(ValueError):
        await canonicalizer.resolve(ExtractedEntity(name="   ", confidence=0.5, evidence="e"))
