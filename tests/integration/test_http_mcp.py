"""Isolated HTTP contract/tenant tests, real storage and fake models only."""

import asyncio
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session as client_session
from pydantic import SecretStr

from multi_rag_harness.config import Settings
from multi_rag_harness.mcp_server.http import HttpSettings, create_http_app
from multi_rag_harness.mcp_server.server import build_container, close_container, create_server
from tests.fakes import FakeCodexClient, FakeEmbedder, FakeReranker
from tests.integration.test_codex_extraction import DOC_TEXT, VALID_RESPONSE
from tests.integration.test_mcp_tools import EXPECTED_TOOLS

TOKEN = "fixture-token-with-at-least-32-characters"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "X-MRH-Site-User-ID": "owner-a",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-11-25",
}


@pytest.fixture
async def http_harness(settings, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "graph.md").write_text(DOC_TEXT, encoding="utf-8")
    (corpus / "search.py").write_text("def hybrid_search(): return 'RRF'", encoding="utf-8")
    built = await build_container(
        settings,
        embedder=FakeEmbedder(32),
        reranker=FakeReranker(),
        codex_client=FakeCodexClient([VALID_RESPONSE]),
    )
    config = HttpSettings(
        backend_token=SecretStr(TOKEN),
        site_user_id="owner-a",
        ingest_root=corpus,
        allowed_host="backend.example",
    )
    app = create_http_app(settings, config, container=built)
    ready, stop = asyncio.Event(), asyncio.Event()

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            ready.set()
            await stop.wait()

    lifecycle = asyncio.create_task(run_lifespan())
    try:
        await asyncio.wait_for(ready.wait(), timeout=10)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="https://backend.example",
            headers=HEADERS,
        ) as client:
            yield client, built, corpus
    finally:
        stop.set()
        await lifecycle
        await close_container(built)


async def rpc(client, method, params=None):
    response = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        },
    )
    assert response.status_code == 200, response.text
    assert "mcp-session-id" not in response.headers
    return response.json()


async def call(client, name, arguments):
    return (await rpc(client, "tools/call", {"name": name, "arguments": arguments}))["result"]


async def test_official_http_client_session(http_harness):
    client, _, corpus = http_harness
    async with streamable_http_client("https://backend.example/mcp", http_client=client) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            assert {tool.name for tool in (await session.list_tools()).tools} == EXPECTED_TOOLS
            result = await session.call_tool("rag_ingest_path", {"path": str(corpus)})
            assert not result.isError
            found = await session.call_tool("rag_search", {"query": "embedded graph store"})
            assert found.structuredContent and found.structuredContent["results"]


async def test_backend_owns_storage_for_process_lifetime(settings, tmp_path, monkeypatch):
    import multi_rag_harness.mcp_server.server as server_module

    monkeypatch.setattr(server_module, "create_embedder", lambda _: FakeEmbedder(32))
    monkeypatch.setattr(server_module, "create_reranker", lambda _: FakeReranker())
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    config = HttpSettings(
        backend_token=SecretStr(TOKEN),
        site_user_id="owner-a",
        ingest_root=corpus,
        allowed_host="backend.example",
    )
    app = create_http_app(settings, config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="https://backend.example", headers=HEADERS
        ) as client:
            await rpc(client, "tools/list")
            await rpc(client, "tools/list")
    # Reopening embedded storage proves the HTTP lifespan released its lock.
    built = await build_container(settings, embedder=FakeEmbedder(32), reranker=FakeReranker())
    await close_container(built)


async def test_http_discovery_matches_all_stdio_contracts(http_harness):
    client, built, _ = http_harness
    initialized = await rpc(
        client,
        "initialize",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "fixture", "version": "1"},
        },
    )
    assert initialized["result"]["protocolVersion"] == "2025-11-25"
    notification = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        },
    )
    assert notification.status_code == 202
    assert not notification.content
    tools = (await rpc(client, "tools/list"))["result"]["tools"]
    local = create_server(built.settings, container=built)
    async with client_session(local._mcp_server) as session:
        expected = (await session.list_tools()).model_dump(mode="json", exclude_none=True)["tools"]
    assert tools == expected
    assert {tool["name"] for tool in tools} == EXPECTED_TOOLS


async def test_all_http_tools_preserve_retrieval_graph_memory_and_provenance(http_harness):
    client, built, corpus = http_harness
    ingest = await call(
        client, "rag_ingest_path", {"path": str(corpus), "options": {"extract": True}}
    )
    assert ingest["structuredContent"]["documents_ingested"] == 2
    assert (await built.extraction.run_pending()).runs_completed >= 1
    for name, payload in {
        "trace": {"task": "fixture retrieval", "outcome": "success"},
        "decision": {"title": "fixture RRF", "decision": "use RRF", "rationale": "fixture"},
        "failure": {"error_text": "fixture error"},
        "tool": {"server": "fixture", "name": "search", "description": "fixture search"},
    }.items():
        stored = await call(client, f"memory_store_{name}", {"payload": payload})
        assert stored["structuredContent"]["record_id"]
    local = create_server(built.settings, container=built)
    async with client_session(local._mcp_server) as session:
        for name in [
            "rag_search",
            "trace_search",
            "decision_search",
            "error_search",
            "code_search",
            "tool_search",
        ]:
            arguments = {"query": "fixture RRF embedded graph store", "top_k": 3, "rerank": True}
            remote = await call(client, name, arguments)
            expected = await session.call_tool(name, arguments)
            assert remote["structuredContent"] == expected.structuredContent
            assert not remote["isError"]
        search = await call(
            client, "rag_search", {"query": "embedded graph store", "graph_expand": True}
        )
        hit = search["structuredContent"]["results"][0]
        assert hit["source_id"] and hit["source_path"] and hit["excerpt"]
        entities = await call(client, "graph_search_entities", {"query": "Kuzu"})
        entity_id = entities["structuredContent"]["results"][0]["id"]
        for name, args in {
            "rag_get_source": {"source_id": hit["id"], "around": 2},
            "graph_search_entities": {"query": "Kuzu", "top_k": 3},
            "graph_search_claims": {"query": "embedded graph store", "top_k": 3},
            "graph_expand": {"entity_id": entity_id, "depth": 2},
            "graph_get_sources": {"graph_item_id": entity_id},
        }.items():
            remote = await call(client, name, args)
            expected = await session.call_tool(name, args)
            assert remote["structuredContent"] == expected.structuredContent


@pytest.mark.parametrize(
    "method,params",
    [
        ("tools/list", {}),
        ("tools/call", {"name": "rag_get_source", "arguments": {"source_id": "other-tenant-id"}}),
        ("tools/call", {"name": "graph_expand", "arguments": {"entity_id": "other-tenant-id"}}),
        (
            "tools/call",
            {"name": "memory_store_trace", "arguments": {"payload": {"task": "x", "outcome": "x"}}},
        ),
    ],
)
async def test_every_operation_checks_service_and_tenant_identity(http_harness, method, params):
    client, _, _ = http_harness
    for headers, status in [
        ({"Authorization": ""}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"X-MRH-Site-User-ID": ""}, 403),
        ({"X-MRH-Site-User-ID": "owner-b"}, 403),
    ]:
        response = await client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params,
            },
        )
        assert response.status_code == status
        assert "other-tenant-id" not in response.text


async def test_protocol_tool_errors_and_corpus_confinement(http_harness, tmp_path):
    client, _, corpus = http_harness
    for path in [str(tmp_path), str(corpus / ".."), str(tmp_path / "secret.md")]:
        result = await call(client, "rag_ingest_path", {"path": path})
        assert result["isError"]
        assert "configured corpus root" in result["content"][0]["text"]
    assert (await call(client, "rag_get_source", {"source_id": "missing"}))["isError"]
    assert (await call(client, "rag_search", {}))["isError"]
    unknown = await rpc(client, "unknown/method")
    assert unknown["error"]["code"] == -32602  # Pinned SDK rejects its request union.
    for headers, status in [
        ({"Origin": "https://attacker.example"}, 403),
        ({"Host": "attacker.example"}, 421),
        ({"MCP-Protocol-Version": "unsupported"}, 400),
    ]:
        response = await client.post(
            "/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        assert response.status_code == status
    assert (await client.get("/mcp")).status_code == 405
    assert (await client.post("/other")).status_code == 404
    assert (await client.post("/mcp", content="x" * (1024 * 1024 + 1))).status_code == 413
    assert (
        await client.post("/mcp", content="invalid", headers={"Content-Type": "application/json"})
    ).status_code == 400


def test_http_config_fails_closed(tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HttpSettings(
            backend_token=SecretStr("short"),
            site_user_id="a",
            ingest_root=tmp_path,
            allowed_host="x",
        )
    with pytest.raises(ValidationError):
        HttpSettings(
            backend_token=SecretStr(TOKEN),
            site_user_id="a",
            ingest_root=Path("relative"),
            allowed_host="x",
        )
    config = HttpSettings(
        backend_token=SecretStr(TOKEN), site_user_id="a", ingest_root=tmp_path, allowed_host="x"
    )
    with pytest.raises(ValueError, match="outside"):
        create_http_app(Settings(data_dir=tmp_path / "data"), config)
