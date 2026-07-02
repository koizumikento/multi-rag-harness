"""Integration tests for the SQLite metadata + keyword store."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from multi_rag_harness.storage.interfaces import (
    AliasRecord,
    ChunkRecord,
    ClaimRecord,
    DecisionRecord,
    DocumentRecord,
    ExtractionRunRecord,
    FailureRecord,
    ProvenanceRecord,
    SearchFilters,
    ToolRecord,
    TraceRecord,
    utcnow,
)
from multi_rag_harness.storage.sqlite import SqliteStore, build_fts_match


@pytest.fixture
async def store(tmp_path: Path):
    store = SqliteStore(tmp_path / "test.db")
    await store.initialize()
    yield store
    await store.close()


def make_document(**overrides) -> DocumentRecord:
    defaults: dict[str, Any] = dict(
        source_uri="/repo/docs/example.md",
        title="Example Doc",
        content_hash="hash-1",
    )
    defaults.update(overrides)
    return DocumentRecord(**defaults)


def make_chunk(document: DocumentRecord, ordinal: int = 0, **overrides) -> ChunkRecord:
    defaults: dict[str, Any] = dict(
        document_id=document.id,
        ordinal=ordinal,
        text=f"chunk text {ordinal}",
        token_count=10,
        scope=document.scope,
        kind=document.kind,
        source_type=document.source_type,
    )
    defaults.update(overrides)
    return ChunkRecord(**defaults)


async def index_document(store: SqliteStore, doc: DocumentRecord, chunks: list[ChunkRecord]):
    await store.upsert_document(doc)
    await store.insert_chunks(chunks)
    await store.index_chunks(chunks, doc.title)


async def test_initialize_is_idempotent(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "idempotent.db")
    await store.initialize()
    await store.close()
    store2 = SqliteStore(tmp_path / "idempotent.db")
    await store2.initialize()
    await store2.close()


async def test_document_roundtrip_and_uri_lookup(store: SqliteStore) -> None:
    doc = make_document(tags=["a", "b"], metadata={"k": "v"})
    await store.upsert_document(doc)
    loaded = await store.get_document(doc.id)
    assert loaded is not None
    assert loaded.source_uri == doc.source_uri
    assert loaded.tags == ["a", "b"]
    assert loaded.metadata == {"k": "v"}

    by_uri = await store.find_document_by_uri(doc.source_uri, "default")
    assert by_uri is not None and by_uri.id == doc.id
    assert await store.find_document_by_uri(doc.source_uri, "other") is None

    doc.title = "Updated"
    await store.upsert_document(doc)
    reloaded = await store.get_document(doc.id)
    assert reloaded is not None and reloaded.title == "Updated"


async def test_chunk_window(store: SqliteStore) -> None:
    doc = make_document()
    chunks = [make_chunk(doc, i) for i in range(5)]
    await index_document(store, doc, chunks)

    window = await store.get_chunk_window(chunks[2].id, around=1)
    assert [c.ordinal for c in window] == [1, 2, 3]

    window = await store.get_chunk_window(chunks[0].id, around=2)
    assert [c.ordinal for c in window] == [0, 1, 2]

    assert await store.get_chunk_window("missing", around=1) == []


async def test_keyword_search_and_filters(store: SqliteStore) -> None:
    doc_a = make_document(source_uri="/a.md", kind="doc", scope="default", tags=["alpha"])
    doc_b = make_document(
        source_uri="/b.py", kind="code", scope="proj", language="python", tags=["beta"]
    )
    chunk_a = make_chunk(doc_a, 0, text="ImportError cannot import name FooBar", tags=["alpha"])
    chunk_b = make_chunk(
        doc_b,
        0,
        text="def foobar(): raise ImportError",
        language="python",
        path="src/b.py",
        tags=["beta"],
    )
    await index_document(store, doc_a, [chunk_a])
    await index_document(store, doc_b, [chunk_b])

    results = await store.search_chunks("ImportError", None, 10)
    assert {r.id for r in results} == {chunk_a.id, chunk_b.id}

    results = await store.search_chunks("ImportError", SearchFilters(kinds=["code"]), 10)
    assert [r.id for r in results] == [chunk_b.id]

    results = await store.search_chunks("ImportError", SearchFilters(scopes=["default"]), 10)
    assert [r.id for r in results] == [chunk_a.id]

    results = await store.search_chunks("ImportError", SearchFilters(language="python"), 10)
    assert [r.id for r in results] == [chunk_b.id]

    results = await store.search_chunks("ImportError", SearchFilters(tags=["alpha"]), 10)
    assert [r.id for r in results] == [chunk_a.id]

    results = await store.search_chunks("ImportError", SearchFilters(path_prefix="src/"), 10)
    assert [r.id for r in results] == [chunk_b.id]

    results = await store.search_chunks(
        "ImportError", SearchFilters(created_after=utcnow() + timedelta(days=1)), 10
    )
    assert results == []


async def test_temporal_filtering(store: SqliteStore) -> None:
    doc = make_document()
    expired = make_chunk(
        doc,
        0,
        text="temporal fact expired",
        valid_to=datetime(2020, 1, 1, tzinfo=UTC),
    )
    current = make_chunk(doc, 1, text="temporal fact current")
    await index_document(store, doc, [expired, current])

    results = await store.search_chunks("temporal", None, 10)
    assert [r.id for r in results] == [current.id]

    results = await store.search_chunks("temporal", SearchFilters(include_expired=True), 10)
    assert {r.id for r in results} == {expired.id, current.id}

    # Point-in-time query before expiry sees the old fact.
    results = await store.search_chunks(
        "temporal", SearchFilters(valid_at=datetime(2019, 6, 1, tzinfo=UTC)), 10
    )
    assert {r.id for r in results} == {expired.id, current.id}


async def test_expire_chunks_for_document(store: SqliteStore) -> None:
    doc = make_document()
    chunk = make_chunk(doc, 0, text="soon to be superseded decision")
    await index_document(store, doc, [chunk])

    await store.expire_chunks_for_document(doc.id, utcnow() - timedelta(seconds=1))
    assert await store.search_chunks("superseded", None, 10) == []
    results = await store.search_chunks("superseded", SearchFilters(include_expired=True), 10)
    assert [r.id for r in results] == [chunk.id]


async def test_delete_chunks_removes_fts_and_returns_embedding_ids(store: SqliteStore) -> None:
    doc = make_document()
    chunks = [
        make_chunk(doc, 0, text="unique deletable text", embedding_id="emb-0"),
        make_chunk(doc, 1, text="unique deletable text two", embedding_id="emb-1"),
    ]
    await index_document(store, doc, chunks)

    embedding_ids = await store.delete_chunks_for_document(doc.id)
    assert sorted(embedding_ids) == ["emb-0", "emb-1"]
    assert await store.search_chunks("deletable", None, 10) == []
    assert await store.get_chunks_for_document(doc.id) == []


async def test_delete_document_cascades(store: SqliteStore) -> None:
    doc = make_document()
    chunk = make_chunk(doc, 0, text="cascade target text")
    await index_document(store, doc, [chunk])

    await store.delete_document(doc.id)
    assert await store.get_document(doc.id) is None
    assert await store.get_chunk(chunk.id) is None
    assert await store.search_chunks("cascade", None, 10) == []


async def test_graph_items_fts(store: SqliteStore) -> None:
    await store.index_graph_item("ent-1", "entity", "Qdrant vector database")
    await store.index_graph_item("claim-1", "claim", "Qdrant supports local embedded mode")

    entities = await store.search_graph_items("Qdrant", "entity", 10)
    assert [r.id for r in entities] == ["ent-1"]
    claims = await store.search_graph_items("Qdrant", "claim", 10)
    assert [r.id for r in claims] == ["claim-1"]

    # Re-index replaces the old row instead of duplicating it.
    await store.index_graph_item("ent-1", "entity", "renamed graph store")
    assert await store.search_graph_items("Qdrant", "entity", 10) == []
    assert [r.id for r in await store.search_graph_items("renamed", "entity", 10)] == ["ent-1"]

    await store.remove_graph_item("ent-1")
    assert await store.search_graph_items("renamed", "entity", 10) == []


async def test_provenance_and_entity_ids_for_chunks(store: SqliteStore) -> None:
    doc = make_document()
    chunk = make_chunk(doc, 0)
    await index_document(store, doc, [chunk])
    await store.insert_provenance(
        [
            ProvenanceRecord(
                item_type="entity",
                item_id="ent-1",
                source_id=doc.id,
                chunk_id=chunk.id,
                evidence_text="evidence",
            ),
            ProvenanceRecord(
                item_type="claim",
                item_id="claim-1",
                source_id=doc.id,
                chunk_id=chunk.id,
            ),
        ]
    )
    rows = await store.get_provenance_for_item("ent-1")
    assert len(rows) == 1 and rows[0].evidence_text == "evidence"

    entity_ids = await store.get_entity_ids_for_chunks([chunk.id])
    assert entity_ids == ["ent-1"]


async def test_extraction_run_claim_is_atomic(store: SqliteStore) -> None:
    runs = [
        ExtractionRunRecord(source_id="doc", chunk_id=f"chunk-{i}", prompt_version="v1")
        for i in range(3)
    ]
    await store.create_extraction_runs(runs)

    claimed = await store.claim_pending_extraction_runs(2)
    assert len(claimed) == 2
    assert all(r.status == "running" and r.started_at is not None for r in claimed)

    remaining = await store.claim_pending_extraction_runs(5)
    assert len(remaining) == 1
    assert await store.claim_pending_extraction_runs(5) == []

    await store.update_extraction_run(claimed[0].id, status="completed", codex_thread_id="thread-1")
    await store.update_extraction_run(claimed[1].id, status="failed", error="boom")


async def test_memory_tables_roundtrip(store: SqliteStore) -> None:
    trace = TraceRecord(task="fix bug", outcome="success", tools_used=["pytest"])
    await store.insert_trace(trace)
    loaded_trace = await store.get_trace(trace.id)
    assert loaded_trace is not None and loaded_trace.tools_used == ["pytest"]

    decision = DecisionRecord(title="Use SQLite", decision="sqlite for metadata")
    await store.insert_decision(decision)
    loaded_decision = await store.get_decision(decision.id)
    assert loaded_decision is not None and loaded_decision.status == "accepted"

    failure = FailureRecord(error_text="ImportError: no module named foo")
    await store.insert_failure(failure)
    loaded_failure = await store.get_failure(failure.id)
    assert loaded_failure is not None and "ImportError" in loaded_failure.error_text


async def test_decision_supersede(store: SqliteStore) -> None:
    old = DecisionRecord(title="Old", decision="old way")
    new = DecisionRecord(title="New", decision="new way", supersedes=old.id)
    await store.insert_decision(old)
    await store.insert_decision(new)
    await store.mark_decision_superseded(old.id, new.id)

    loaded = await store.get_decision(old.id)
    assert loaded is not None
    assert loaded.status == "superseded"
    assert loaded.superseded_by == new.id


async def test_tool_record_upsert(store: SqliteStore) -> None:
    record = ToolRecord(server="mrh", name="rag_search", description="hybrid search")
    stored = await store.upsert_tool_record(record)
    assert stored.id == record.id

    replacement = ToolRecord(server="mrh", name="rag_search", description="updated desc")
    stored2 = await store.upsert_tool_record(replacement)
    assert stored2.id == record.id  # keeps original identity
    loaded = await store.get_tool_record(record.id)
    assert loaded is not None and loaded.description == "updated desc"


async def test_claims_and_aliases(store: SqliteStore) -> None:
    claim = ClaimRecord(
        subject_entity_id="ent-1",
        predicate="uses",
        object_text="Kuzu",
        modality="fact",
        confidence=0.9,
    )
    expired_claim = ClaimRecord(
        subject_entity_id="ent-1",
        predicate="used",
        object_text="Neo4j",
        modality="fact",
        confidence=0.8,
        valid_to=datetime(2020, 1, 1, tzinfo=UTC),
    )
    await store.insert_claims([claim, expired_claim])
    assert (await store.get_claim(claim.id)) is not None

    current = await store.get_claims_for_entity("ent-1")
    assert [c.id for c in current] == [claim.id]
    past = await store.get_claims_for_entity("ent-1", valid_at=datetime(2019, 1, 1, tzinfo=UTC))
    assert {c.id for c in past} == {claim.id, expired_claim.id}

    aliases = [
        AliasRecord(entity_id="ent-1", alias="Kuzu DB", normalized_alias="kuzu db"),
        AliasRecord(entity_id="ent-1", alias="KuzuDB", normalized_alias="kuzudb"),
    ]
    await store.insert_aliases(aliases)
    # Duplicate insert is ignored.
    await store.insert_aliases(
        [AliasRecord(entity_id="ent-1", alias="Kuzu DB", normalized_alias="kuzu db")]
    )
    stored_aliases = await store.get_aliases_for_entity("ent-1")
    assert len(stored_aliases) == 2
    assert await store.find_entity_id_by_normalized_alias("kuzudb") == "ent-1"
    assert await store.find_entity_id_by_normalized_alias("unknown") is None


def test_build_fts_match_basic() -> None:
    assert build_fts_match("hello world") == '"hello" OR "world"'


def test_build_fts_match_empty_raises() -> None:
    with pytest.raises(ValueError):
        build_fts_match("   ")
