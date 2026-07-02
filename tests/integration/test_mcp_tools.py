"""In-process MCP server tests: list tools, call tools, check structured output."""

import json
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session as client_session

from multi_rag_harness.config import Settings
from multi_rag_harness.mcp_server.server import (
    AppContainer,
    build_container,
    close_container,
    create_server,
)
from tests.conftest import FAKE_DIMENSION
from tests.fakes import FakeCodexClient, FakeEmbedder, FakeReranker
from tests.integration.test_codex_extraction import DOC_TEXT, VALID_RESPONSE

EXPECTED_TOOLS = {
    "rag_ingest_path",
    "rag_search",
    "rag_get_source",
    "graph_search_entities",
    "graph_search_claims",
    "graph_expand",
    "graph_get_sources",
    "trace_search",
    "decision_search",
    "error_search",
    "code_search",
    "tool_search",
    "memory_store_trace",
    "memory_store_decision",
    "memory_store_failure",
}


def sc(result) -> dict:
    """Unwrap structured content, asserting it exists."""
    assert result.structuredContent is not None
    return result.structuredContent


@pytest.fixture
async def container(settings: Settings):
    container = await build_container(
        settings,
        embedder=FakeEmbedder(FAKE_DIMENSION),
        reranker=FakeReranker(),
        codex_client=FakeCodexClient([VALID_RESPONSE]),
    )
    yield container
    await close_container(container)


@pytest.fixture
def server(settings: Settings, container: AppContainer):
    return create_server(settings, container=container)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "graph.md").write_text(DOC_TEXT, encoding="utf-8")
    (root / "search.py").write_text(
        "def hybrid_search(query):\n    return rrf_merge(query)\n", encoding="utf-8"
    )
    return root


async def test_list_tools_matches_spec_surface(server) -> None:
    async with client_session(server._mcp_server) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        assert names == EXPECTED_TOOLS


async def test_ingest_search_get_source_flow(server, corpus: Path) -> None:
    async with client_session(server._mcp_server) as client:
        ingest = await client.call_tool("rag_ingest_path", {"path": str(corpus)})
        assert not ingest.isError
        assert sc(ingest)["documents_ingested"] == 2

        search = await client.call_tool(
            "rag_search",
            {"query": "embedded graph store", "rerank": False},
        )
        assert not search.isError
        payload = sc(search)
        assert payload["reranked"] is False
        assert payload["results"], "expected search hits"
        top = payload["results"][0]
        assert set(top) >= {
            "id",
            "kind",
            "score",
            "source_id",
            "source_path",
            "title",
            "excerpt",
            "metadata",
        }

        # Chunk id → windowed context.
        source = await client.call_tool("rag_get_source", {"source_id": top["id"]})
        assert not source.isError
        chunks = sc(source)["chunks"]
        assert any(c["is_target"] for c in chunks)
        assert sc(source)["source_path"].endswith("graph.md")

        # Document id → document head.
        doc_source = await client.call_tool(
            "rag_get_source", {"source_id": top["source_id"], "around": 2}
        )
        assert not doc_source.isError
        assert sc(doc_source)["source_id"] == top["source_id"]

        missing = await client.call_tool("rag_get_source", {"source_id": "nope"})
        assert missing.isError


async def test_code_search_tool(server, corpus: Path) -> None:
    async with client_session(server._mcp_server) as client:
        await client.call_tool("rag_ingest_path", {"path": str(corpus)})
        result = await client.call_tool(
            "code_search", {"query": "hybrid_search rrf_merge", "rerank": False}
        )
        assert not result.isError
        results = sc(result)["results"]
        assert results and all(r["kind"] == "code" for r in results)


async def test_memory_store_and_search_tools(server) -> None:
    async with client_session(server._mcp_server) as client:
        stored = await client.call_tool(
            "memory_store_decision",
            {
                "payload": {
                    "title": "Use RRF",
                    "decision": "Merge keyword and vector results with RRF k=60",
                    "rationale": "backend agnostic",
                }
            },
        )
        assert not stored.isError
        record_id = sc(stored)["record_id"]

        found = await client.call_tool(
            "decision_search", {"query": "RRF keyword vector merge", "rerank": False}
        )
        assert not found.isError
        results = sc(found)["results"]
        assert any(r["metadata"].get("record_id") == record_id for r in results)

        trace = await client.call_tool(
            "memory_store_trace",
            {"payload": {"task": "wire MCP server", "outcome": "success"}},
        )
        assert not trace.isError

        failure = await client.call_tool(
            "memory_store_failure",
            {"payload": {"error_text": "RuntimeError: lifespan not initialized"}},
        )
        assert not failure.isError

        errors = await client.call_tool(
            "error_search", {"query": "lifespan not initialized", "rerank": False}
        )
        assert not errors.isError
        assert sc(errors)["results"]

        traces = await client.call_tool(
            "trace_search", {"query": "wire MCP server", "rerank": False}
        )
        assert not traces.isError
        assert sc(traces)["results"]


async def test_graph_tools_flow(server, container: AppContainer, corpus: Path) -> None:
    async with client_session(server._mcp_server) as client:
        await client.call_tool(
            "rag_ingest_path",
            {"path": str(corpus / "graph.md"), "options": {"extract": True}},
        )
        # Extraction is a batch job outside the MCP request path.
        summary = await container.extraction.run_pending()
        assert summary.runs_completed == 1

        entities = await client.call_tool("graph_search_entities", {"query": "Kuzu"})
        assert not entities.isError
        entity_results = sc(entities)["results"]
        assert entity_results and entity_results[0]["kind"] == "entity"
        entity_id = entity_results[0]["id"]

        claims = await client.call_tool("graph_search_claims", {"query": "embedded graph store"})
        assert not claims.isError
        assert sc(claims)["results"]

        expanded = await client.call_tool("graph_expand", {"entity_id": entity_id, "depth": 1})
        assert not expanded.isError
        names = {e["canonical_name"] for e in sc(expanded)["entities"]}
        assert names == {"Kuzu", "GraphRAG"}
        assert sc(expanded)["relations"]

        sources = await client.call_tool("graph_get_sources", {"graph_item_id": entity_id})
        assert not sources.isError
        source_rows = sc(sources)["sources"]
        assert source_rows and source_rows[0]["evidence_text"]

        # rag_search graph_expand appends linked entities to the tail.
        enriched = await client.call_tool(
            "rag_search",
            {"query": "embedded graph store", "rerank": False, "graph_expand": True},
        )
        assert not enriched.isError
        kinds = [r["kind"] for r in sc(enriched)["results"]]
        assert "entity" in kinds


async def test_codex_model_setting_reaches_sdk_client(settings: Settings) -> None:
    from multi_rag_harness.codex.client import CodexSdkClient

    settings.codex.model = "gpt-5-codex"
    built = await build_container(
        settings, embedder=FakeEmbedder(FAKE_DIMENSION), reranker=FakeReranker()
    )
    try:
        assert isinstance(built.codex_client, CodexSdkClient)
        assert built.codex_client._model == "gpt-5-codex"
    finally:
        await close_container(built)


async def test_tool_search_over_stored_tool_records(server, container: AppContainer) -> None:
    from multi_rag_harness.memory.tools import ToolRecordPayload

    await container.tools_memory.store(
        ToolRecordPayload(
            server="multi-rag-harness",
            name="rag_search",
            description="hybrid retrieval with rerank",
            input_schema=json.loads('{"type": "object"}'),
        )
    )
    async with client_session(server._mcp_server) as client:
        result = await client.call_tool(
            "tool_search", {"query": "hybrid retrieval rerank", "rerank": False}
        )
        assert not result.isError
        results = sc(result)["results"]
        assert results and results[0]["kind"] == "tool"
