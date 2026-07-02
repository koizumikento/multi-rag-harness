"""Integration tests for the embedded Qdrant vector index."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from multi_rag_harness.storage.interfaces import (
    SearchFilters,
    VectorDimensionMismatchError,
    VectorPoint,
)
from multi_rag_harness.storage.qdrant import (
    QdrantVectorIndex,
    build_point_payload,
    client_kwargs,
)

DIM = 8


@pytest.fixture
async def index(tmp_path: Path):
    index = QdrantVectorIndex(tmp_path / "qdrant", "test_vectors")
    await index.initialize(DIM)
    yield index
    await index.close()


def vec(direction: int) -> list[float]:
    values = [0.0] * DIM
    values[direction] = 1.0
    return values


def make_point(direction: int, **payload_overrides) -> VectorPoint:
    item_id = payload_overrides.pop("item_id", f"item-{direction}")
    payload = build_point_payload(item_id=item_id, item_kind="chunk", **payload_overrides)
    return VectorPoint(id=str(uuid.uuid4()), vector=vec(direction), payload=payload)


async def test_upsert_and_search_order(index: QdrantVectorIndex) -> None:
    await index.upsert([make_point(0), make_point(1), make_point(2)])
    results = await index.search(vec(0), None, limit=2)
    assert results[0].id == "item-0"
    assert results[0].score > results[1].score


async def test_returns_item_id_not_point_id(index: QdrantVectorIndex) -> None:
    point = make_point(0, item_id="the-chunk-id")
    await index.upsert([point])
    results = await index.search(vec(0), None, limit=1)
    assert results[0].id == "the-chunk-id"
    assert results[0].id != point.id


async def test_filters(index: QdrantVectorIndex) -> None:
    await index.upsert(
        [
            make_point(0, item_id="a", scope="s1", kind="doc", tags=["x"]),
            make_point(1, item_id="b", scope="s2", kind="code", tags=["y"], path="src/b.py"),
        ]
    )
    results = await index.search(vec(0), SearchFilters(scopes=["s1"]), limit=10)
    assert [r.id for r in results] == ["a"]

    results = await index.search(vec(0), SearchFilters(kinds=["code"]), limit=10)
    assert [r.id for r in results] == ["b"]

    results = await index.search(vec(0), SearchFilters(tags=["y"]), limit=10)
    assert [r.id for r in results] == ["b"]

    results = await index.search(vec(0), SearchFilters(path_prefix="src/"), limit=10)
    assert [r.id for r in results] == ["b"]


async def test_temporal_filter(index: QdrantVectorIndex) -> None:
    await index.upsert(
        [
            make_point(0, item_id="expired", valid_to=datetime(2020, 1, 1, tzinfo=UTC)),
            make_point(0, item_id="current"),
        ]
    )
    results = await index.search(vec(0), None, limit=10)
    assert [r.id for r in results] == ["current"]

    results = await index.search(vec(0), SearchFilters(include_expired=True), limit=10)
    assert {r.id for r in results} == {"expired", "current"}

    results = await index.search(
        vec(0), SearchFilters(valid_at=datetime(2019, 1, 1, tzinfo=UTC)), limit=10
    )
    assert {r.id for r in results} == {"expired", "current"}


async def test_item_kinds_filter(index: QdrantVectorIndex) -> None:
    chunk = make_point(0, item_id="chunk-1")
    entity_payload = build_point_payload(item_id="entity-1", item_kind="entity")
    entity = VectorPoint(id=str(uuid.uuid4()), vector=vec(0), payload=entity_payload)
    await index.upsert([chunk, entity])

    results = await index.search(vec(0), None, limit=10)  # defaults to chunks
    assert [r.id for r in results] == ["chunk-1"]

    results = await index.search(vec(0), None, limit=10, item_kinds=["entity"])
    assert [r.id for r in results] == ["entity-1"]


async def test_delete(index: QdrantVectorIndex) -> None:
    point = make_point(0, item_id="doomed")
    await index.upsert([point])
    await index.delete([point.id])
    assert await index.search(vec(0), None, limit=10) == []


def test_client_kwargs_embedded_vs_remote(tmp_path: Path) -> None:
    assert client_kwargs(tmp_path / "qdrant") == {"path": str(tmp_path / "qdrant")}
    assert client_kwargs(tmp_path, url="https://qdrant.example.com:6333") == {
        "url": "https://qdrant.example.com:6333"
    }
    assert client_kwargs(tmp_path, url="https://q.example.com", api_key="secret") == {
        "url": "https://q.example.com",
        "api_key": "secret",
    }


async def test_dimension_mismatch_fails_fast(tmp_path: Path) -> None:
    first = QdrantVectorIndex(tmp_path / "qdrant", "dim_check")
    await first.initialize(DIM)
    await first.close()

    second = QdrantVectorIndex(tmp_path / "qdrant", "dim_check")
    with pytest.raises(VectorDimensionMismatchError, match="dimension"):
        await second.initialize(DIM * 2)

    # Matching dimension still opens fine.
    third = QdrantVectorIndex(tmp_path / "qdrant", "dim_check")
    await third.initialize(DIM)
    await third.close()
