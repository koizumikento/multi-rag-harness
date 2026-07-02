"""Integration tests for memory store/search services."""

import pytest

from multi_rag_harness.memory.decisions import DecisionMemoryService, DecisionPayload
from multi_rag_harness.memory.failures import FailureMemoryService, FailurePayload
from multi_rag_harness.memory.tools import ToolMemoryService, ToolRecordPayload
from multi_rag_harness.memory.traces import TraceMemoryService, TracePayload
from multi_rag_harness.storage.interfaces import SearchFilters
from tests.conftest import Harness


@pytest.fixture
def traces(harness: Harness) -> TraceMemoryService:
    return TraceMemoryService(harness.storage.metadata, harness.pipeline, harness.search)


@pytest.fixture
def decisions(harness: Harness) -> DecisionMemoryService:
    return DecisionMemoryService(harness.storage.metadata, harness.pipeline, harness.search)


@pytest.fixture
def failures(harness: Harness) -> FailureMemoryService:
    return FailureMemoryService(harness.storage.metadata, harness.pipeline, harness.search)


@pytest.fixture
def tools(harness: Harness) -> ToolMemoryService:
    return ToolMemoryService(harness.storage.metadata, harness.pipeline, harness.search)


async def test_trace_store_and_search(harness: Harness, traces: TraceMemoryService) -> None:
    ref = await traces.store(
        TracePayload(
            task="migrate storage layer to qdrant",
            outcome="success",
            tools_used=["pytest", "ruff"],
        )
    )
    record = await harness.storage.metadata.get_trace(ref.record_id)
    assert record is not None and record.outcome == "success"

    output = await traces.search("migrate storage qdrant", rerank=False)
    assert output.results
    top = output.results[0]
    assert top.kind == "trace"
    assert top.source_id == ref.document_id
    assert top.metadata["record_id"] == ref.record_id
    assert top.source_path == f"memory://trace/{ref.record_id}"


async def test_decision_store_and_search(
    harness: Harness, decisions: DecisionMemoryService
) -> None:
    ref = await decisions.store(
        DecisionPayload(
            title="Adopt RRF for hybrid merge",
            decision="Use reciprocal rank fusion with k=60",
            rationale="simple and backend agnostic",
        )
    )
    output = await decisions.search("reciprocal rank fusion", rerank=False)
    assert output.results
    assert output.results[0].kind == "decision"
    assert output.results[0].metadata["record_id"] == ref.record_id


async def test_decision_supersede_hides_old_from_default_search(
    harness: Harness, decisions: DecisionMemoryService
) -> None:
    old = await decisions.store(
        DecisionPayload(
            title="Vector backend choice",
            decision="Use pgvector exclusively for vectors",
        )
    )
    new = await decisions.store(
        DecisionPayload(
            title="Vector backend choice v2",
            decision="Use qdrant embedded exclusively for vectors",
            supersedes=old.record_id,
        )
    )

    old_record = await harness.storage.metadata.get_decision(old.record_id)
    assert old_record is not None
    assert old_record.status == "superseded"
    assert old_record.superseded_by == new.record_id

    default = await decisions.search("pgvector exclusively vectors", rerank=False)
    assert all(r.metadata["record_id"] != old.record_id for r in default.results)

    with_expired = await decisions.search(
        "pgvector exclusively vectors",
        filters=SearchFilters(include_expired=True),
        rerank=False,
    )
    assert any(r.metadata["record_id"] == old.record_id for r in with_expired.results)


async def test_failure_store_and_search(failures: FailureMemoryService) -> None:
    await failures.store(
        FailurePayload(
            error_text="kuzu.RuntimeError: Binder exception: unknown column",
            error_category="query",
            fix_applied="aliased duplicate return columns",
        )
    )
    output = await failures.search("Binder exception unknown column", rerank=False)
    assert output.results
    assert output.results[0].kind == "error"
    assert "Binder exception" in output.results[0].excerpt


async def test_tool_store_upserts_and_search(harness: Harness, tools: ToolMemoryService) -> None:
    first = await tools.store(
        ToolRecordPayload(server="mrh", name="rag_search", description="hybrid document search")
    )
    second = await tools.store(
        ToolRecordPayload(
            server="mrh", name="rag_search", description="hybrid search with rerank support"
        )
    )
    assert second.record_id == first.record_id  # upsert on (server, name)
    assert second.document_id == first.document_id  # stable memory document

    output = await tools.search("rerank support", rerank=False)
    assert output.results
    assert output.results[0].kind == "tool"
    # Old description is replaced, not duplicated.
    ids = [r.id for r in output.results]
    assert len(ids) == len(set(ids))
