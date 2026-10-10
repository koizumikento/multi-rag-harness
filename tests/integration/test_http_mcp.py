"""Isolated HTTP contract/tenant tests, real storage and fake models only."""

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session as client_session
from pydantic import SecretStr

from multi_rag_harness.config import Settings
from multi_rag_harness.mcp_server.http import HttpSettings, create_http_app
from multi_rag_harness.mcp_server.modern_http import (
    CAPABILITIES_KEY,
    MODERN_VERSION,
    PROTOCOL_KEY,
    SERVER_INFO_KEY,
    SUPPORTED_VERSIONS,
)
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


async def modern_rpc(client, method, params=None, *, headers=None):
    params = dict(params or {})
    params.setdefault(
        "_meta",
        {
            PROTOCOL_KEY: MODERN_VERSION,
            CAPABILITIES_KEY: {},
            "io.modelcontextprotocol/clientInfo": {"name": "fixture-modern", "version": "1"},
        },
    )
    envelope = {
        "MCP-Protocol-Version": MODERN_VERSION,
        "Mcp-Method": method,
        **({"Mcp-Name": params["name"]} if "name" in params else {}),
        **(headers or {}),
    }
    return await client.post(
        "/mcp",
        headers=envelope,
        json={
            "jsonrpc": "2.0",
            "id": "modern-fixture",
            "method": method,
            "params": params,
        },
    )


async def test_sites_bootstrap_discovery_and_full_modern_discovery(http_harness):
    client, built, _ = http_harness
    for response in [
        await client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": "bootstrap", "method": "server/discover", "params": {}},
        ),
        await modern_rpc(client, "server/discover"),
    ]:
        assert response.status_code == 200
        assert "mcp-session-id" not in response.headers
        payload = response.json()
        assert "error" not in payload
        result = payload["result"]
        assert result["supportedVersions"] == SUPPORTED_VERSIONS
        assert result["capabilities"] == {"tools": {}}
        assert result["resultType"] == "complete"
        assert result["cacheScope"] == "private" and result["ttlMs"] == 0
        assert response.headers["cache-control"] == "no-store"
        assert result["_meta"][SERVER_INFO_KEY]["name"] == built.settings.mcp.server_name
        assert result["_meta"][SERVER_INFO_KEY]["version"]
        assert result["instructions"] == built.settings.mcp.instructions


async def test_modern_tools_use_real_legacy_handlers_and_keep_results(http_harness):
    client, built, corpus = http_harness
    listed = await modern_rpc(client, "tools/list")
    assert listed.status_code == 200
    result = listed.json()["result"]
    assert result["tools"] == (await rpc(client, "tools/list"))["result"]["tools"]
    assert {tool["name"] for tool in result["tools"]} == EXPECTED_TOOLS
    assert result["resultType"] == "complete"
    assert result["cacheScope"] == "private" and result["ttlMs"] == 0
    assert listed.headers["cache-control"] == "no-store"
    # No initialize handshake; the real v1 SDK creates the lifespan/tool context.
    ingested = await modern_rpc(
        client,
        "tools/call",
        {
            "name": "rag_ingest_path",
            "arguments": {"path": str(corpus), "options": {"extract": True}},
        },
    )
    assert ingested.json()["result"]["structuredContent"]["documents_ingested"] == 2
    args = {"query": "embedded graph store", "rerank": True, "top_k": 3}
    found = await modern_rpc(client, "tools/call", {"name": "rag_search", "arguments": args})
    modern = found.json()["result"]
    legacy = await call(client, "rag_search", args)
    assert modern["structuredContent"] == legacy["structuredContent"]
    hit = modern["structuredContent"]["results"][0]
    assert hit["source_id"] and hit["source_path"] and hit["excerpt"]
    await built.extraction.run_pending()
    entities = await modern_rpc(
        client, "tools/call", {"name": "graph_search_entities", "arguments": {"query": "Kuzu"}}
    )
    entity_id = entities.json()["result"]["structuredContent"]["results"][0]["id"]
    for kind, payload in {
        "trace": {"task": "fixture retrieval", "outcome": "success"},
        "decision": {"title": "fixture RRF", "decision": "use RRF", "rationale": "fixture"},
        "failure": {"error_text": "fixture error"},
        "tool": {"server": "fixture", "name": "search", "description": "fixture search"},
    }.items():
        stored = await modern_rpc(
            client,
            "tools/call",
            {"name": f"memory_store_{kind}", "arguments": {"payload": payload}},
        )
        assert stored.json()["result"]["structuredContent"]["record_id"]
    read_args = {
        **{
            f"{kind}_search": {"query": "fixture RRF", "rerank": True, "top_k": 3}
            for kind in ["trace", "decision", "error", "code", "tool"]
        },
        "rag_get_source": {"source_id": hit["id"]},
        "graph_search_claims": {"query": "embedded graph store"},
        "graph_expand": {"entity_id": entity_id, "depth": 2},
        "graph_get_sources": {"graph_item_id": entity_id},
    }
    for name, arguments in read_args.items():
        actual = (
            await modern_rpc(client, "tools/call", {"name": name, "arguments": arguments})
        ).json()["result"]
        assert not actual["isError"]
        assert (
            actual["structuredContent"]
            == (await call(client, name, arguments))["structuredContent"]
        )
    missing = await modern_rpc(
        client, "tools/call", {"name": "rag_get_source", "arguments": {"source_id": "missing"}}
    )
    assert missing.json()["result"]["isError"]
    assert missing.json()["result"]["resultType"] == "complete"
    keyless = await modern_rpc(
        client,
        "tools/call",
        {"name": "graph_get_sources", "arguments": {"graph_item_id": "sites-validation-missing"}},
    )
    assert keyless.json()["result"]["structuredContent"] == {
        "graph_item_id": "sites-validation-missing",
        "sources": [],
    }


async def test_modern_invalid_json_envelopes_and_no_capability_carryover(http_harness):
    client, _, _ = http_harness
    for body, code in [
        ("invalid", -32700),
        ("[]", -32600),
        ('{"id":1,"method":"tools/list"}', -32600),
    ]:
        response = await client.post(
            "/mcp",
            content=body,
            headers={"Content-Type": "application/json", "MCP-Protocol-Version": MODERN_VERSION},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == code
    assert (await modern_rpc(client, "server/discover")).status_code == 200
    # A previous discovery never supplies metadata/capabilities to a later call.
    response = await client.post(
        "/mcp",
        headers={"MCP-Protocol-Version": MODERN_VERSION, "Mcp-Method": "tools/list"},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    assert response.status_code == 400 and response.json()["error"]["code"] == -32602
    rejected = await modern_rpc(client, "tools/list", headers={"Accept": "application/json"})
    assert rejected.status_code == 406
    for params in [
        {
            "_meta": {
                PROTOCOL_KEY: MODERN_VERSION,
                CAPABILITIES_KEY: [],
                "io.modelcontextprotocol/clientInfo": {"name": "client"},
            }
        },
        {"_meta": None},
    ]:
        assert (await modern_rpc(client, "tools/list", params)).status_code == 400


@pytest.mark.parametrize(
    "headers,meta,code",
    [
        ({"Mcp-Method": "wrong"}, None, -32020),
        ({"MCP-Protocol-Version": "2025-11-25"}, None, -32020),
        ({"Mcp-Method": ""}, None, -32020),
        ({}, {}, -32602),
        ({}, {PROTOCOL_KEY: MODERN_VERSION}, -32602),
        (
            {"MCP-Protocol-Version": "2025-11-25"},
            {PROTOCOL_KEY: "2025-11-25", CAPABILITIES_KEY: {}},
            -32602,
        ),
        (
            {"MCP-Protocol-Version": "2099-01-01"},
            {PROTOCOL_KEY: "2099-01-01", CAPABILITIES_KEY: {}},
            -32022,
        ),
    ],
)
async def test_modern_metadata_and_header_validation(http_harness, headers, meta, code):
    client, _, _ = http_harness
    response = await modern_rpc(
        client, "tools/list", {"_meta": meta} if meta is not None else {}, headers=headers
    )
    assert response.status_code == 400
    assert "result" not in response.json()
    assert response.json()["error"]["code"] == code
    if code == -32022:
        assert response.json()["error"]["data"] == {
            "supported": SUPPORTED_VERSIONS,
            "requested": "2099-01-01",
        }


async def test_modern_tool_name_header_errors_methods_and_auth(http_harness):
    client, _, _ = http_harness
    params = {"name": "graph_get_sources", "arguments": {"graph_item_id": "fixture"}}
    for name in ["wrong", "", "=?base64?invalid!?="]:
        response = await modern_rpc(client, "tools/call", params, headers={"Mcp-Name": name})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32020
    encoded = "=?base64?Z3JhcGhfZ2V0X3NvdXJjZXM=?="
    assert (
        await modern_rpc(client, "tools/call", params, headers={"Mcp-Name": encoded})
    ).status_code == 200
    unknown = await modern_rpc(client, "unknown/method")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == -32601
    for method, args in [("server/discover", {}), ("tools/list", {}), ("tools/call", params)]:
        for headers, status in [
            ({"Authorization": ""}, 401),
            ({"X-MRH-Site-User-ID": "other"}, 403),
        ]:
            assert (await modern_rpc(client, method, args, headers=headers)).status_code == status
    for headers, status in [
        ({"Host": "attacker.example"}, 421),
        ({"Origin": "https://attacker.example"}, 403),
    ]:
        assert (await modern_rpc(client, "server/discover", headers=headers)).status_code == status


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


async def test_official_sdk2_discovers_lists_and_calls_real_backend(http_harness):
    """Separate SDK environments; only mocked HTTP crosses the process boundary."""
    python = os.environ.get("MRH_MODERN_CLIENT_PYTHON")
    if not python:
        pytest.skip("SDK 2.3.0 client interpreter is configured in CI")
    client, _, _ = http_harness
    program = r"""
import asyncio
import importlib.metadata
import json
import sys
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def handle(request):
    print(json.dumps({"method": request.method, "headers": dict(request.headers),
                      "body": request.content.decode()}), flush=True)
    wire = json.loads(await asyncio.to_thread(sys.stdin.readline))
    return httpx2.Response(wire["status"], headers=wire["headers"],
                           content=wire["body"].encode(), request=request)

async def main():
    assert importlib.metadata.version("mcp") == "2.3.0"
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http:
        async with streamable_http_client("https://backend.example/mcp", http_client=http,
                                          terminate_on_close=False) as streams:
            async with ClientSession(*streams) as session:
                discovered = await session.discover()
                assert "2026-07-28" in discovered.supported_versions
                assert discovered.cache_scope == "private" and discovered.ttl_ms == 0
                listed = await session.list_tools()
                assert listed.cache_scope == "private" and listed.ttl_ms == 0
                result = await session.call_tool("graph_get_sources",
                    {"graph_item_id": "sites-validation-missing"})
                assert not result.is_error
                print(json.dumps({"done": True, "tools": [tool.name for tool in listed.tools],
                                  "result": result.structured_content}), flush=True)

asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        python,
        "-B",
        "-c",
        program,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin and process.stdout and process.stderr
    completed = None
    methods = []
    try:
        async with asyncio.timeout(30):
            while line := await process.stdout.readline():
                wire = json.loads(line)
                if wire.get("done"):
                    completed = wire
                    continue
                methods.append(json.loads(wire["body"])["method"])
                response = await client.request(
                    wire["method"], "/mcp", headers=wire["headers"], content=wire["body"]
                )
                process.stdin.write(
                    (
                        json.dumps(
                            {
                                "status": response.status_code,
                                "headers": dict(response.headers),
                                "body": response.text,
                            }
                        )
                        + "\n"
                    ).encode()
                )
                await process.stdin.drain()
            errors = (await process.stderr.read()).decode()
            assert await process.wait() == 0, errors
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert completed is not None
    assert set(completed["tools"]) == EXPECTED_TOOLS
    assert completed["result"] == {"graph_item_id": "sites-validation-missing", "sources": []}
    assert methods[0] == "server/discover" and methods[-1] == "tools/call"
    assert "tools/list" in methods


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
