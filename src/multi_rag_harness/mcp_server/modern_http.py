"""2026 HTTP envelope compatibility over the unchanged, stateless v1 SDK tools."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS
from mcp.types import ClientCapabilities, Implementation, JSONRPCRequest
from pydantic import BaseModel, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message

MODERN_VERSION = "2026-07-28"
SUPPORTED_VERSIONS = [MODERN_VERSION, *reversed(SUPPORTED_PROTOCOL_VERSIONS)]
PROTOCOL_KEY = "io.modelcontextprotocol/protocolVersion"
CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"
SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"


class RequestMetadata(BaseModel):
    protocol_version: str = Field(alias=PROTOCOL_KEY)
    client_capabilities: ClientCapabilities = Field(alias=CAPABILITIES_KEY)
    client_info: Implementation | None = Field(
        default=None, alias="io.modelcontextprotocol/clientInfo"
    )


def rpc_error(request_id: Any, status: int, code: int, message: str, **data: Any) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if data:
        error["data"] = data
    return JSONResponse(
        {"jsonrpc": "2.0", "id": request_id, "error": error},
        status_code=status,
        headers={"Cache-Control": "no-store"},
    )


def decoded_name(value: str | None) -> str | None:
    if value is not None and value.startswith("=?base64?") and value.endswith("?="):
        try:
            return base64.b64decode(value[9:-2], validate=True).decode("utf-8")
        except (ValueError, UnicodeError, binascii.Error):
            return None
    return value


async def modern_response(
    request: Request, body: bytes, app: ASGIApp, server_info: dict, instructions: str
) -> JSONResponse | None:
    """Return None for legacy traffic; modern calls execute through the v1 SDK.

    Bare server/discover is the Sites bootstrap probe. Only discovery allows
    omitted metadata; tools always require a complete, matching modern envelope.
    """
    header_version = request.headers.get("mcp-protocol-version")
    try:
        message = json.loads(body)
    except (ValueError, UnicodeError):
        if header_version == MODERN_VERSION:
            return rpc_error(None, 400, -32700, "Invalid JSON")
        return None
    if not isinstance(message, dict):
        if header_version == MODERN_VERSION:
            return rpc_error(None, 400, -32600, "Expected a single JSON-RPC request")
        return None
    params = message.get("params", {})
    meta = params.get("_meta", {}) if isinstance(params, dict) else {}
    if (
        message.get("method") != "server/discover"
        and header_version != MODERN_VERSION
        and not (isinstance(meta, dict) and PROTOCOL_KEY in meta)
    ):
        return None
    try:
        rpc = JSONRPCRequest.model_validate(message)
    except ValidationError:
        return rpc_error(None, 400, -32600, "Invalid JSON-RPC request")
    if not isinstance(params, dict):
        return rpc_error(rpc.id, 400, -32602, "params must be an object")
    bootstrap = rpc.method == "server/discover" and not params
    if bootstrap and request.headers.get("mcp-method") not in {None, rpc.method}:
        return rpc_error(rpc.id, 400, -32020, "Method header mismatch")
    if not bootstrap:
        try:
            metadata = RequestMetadata.model_validate(meta)
        except ValidationError:
            return rpc_error(rpc.id, 400, -32602, "Required request metadata is missing or invalid")
        if (
            header_version != metadata.protocol_version
            or request.headers.get("mcp-method") != rpc.method
        ):
            return rpc_error(rpc.id, 400, -32020, "Protocol or method header mismatch")
        if rpc.method == "tools/call" and (
            decoded_name(request.headers.get("mcp-name")) != params.get("name")
            or not isinstance(params.get("name"), str)
        ):
            return rpc_error(rpc.id, 400, -32020, "Tool name header mismatch")
    if header_version is not None and header_version not in SUPPORTED_VERSIONS:
        return rpc_error(
            rpc.id,
            400,
            -32022,
            "Unsupported protocol version",
            supported=SUPPORTED_VERSIONS,
            requested=header_version,
        )
    if not bootstrap and header_version != MODERN_VERSION:
        return rpc_error(
            rpc.id,
            400,
            -32602,
            "Per-request metadata requires protocol version 2026-07-28",
        )
    accept = {part.strip() for part in request.headers.get("accept", "").split(",")}
    if not {"application/json", "text/event-stream"}.issubset(accept):
        return rpc_error(rpc.id, 406, -32600, "Accept must include JSON and event-stream")
    result_meta = {SERVER_INFO_KEY: server_info}
    if rpc.method == "server/discover":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": rpc.id,
                "result": {
                    "supportedVersions": SUPPORTED_VERSIONS,
                    "capabilities": {"tools": {}},
                    "instructions": instructions,
                    "resultType": "complete",
                    "_meta": result_meta,
                },
            },
            headers={"Cache-Control": "no-store"},
        )
    if rpc.method not in {"tools/list", "tools/call"}:
        return rpc_error(rpc.id, 404, -32601, "Method not found")

    # The legacy SDK supplies its normal lifespan, validation and result shaping.
    scope = dict(request.scope)
    scope["headers"] = [
        (key, value)
        for key, value in scope["headers"]
        if key not in {b"mcp-protocol-version", b"content-length"}
    ] + [(b"mcp-protocol-version", b"2025-11-25")]
    events: list[Message] = []

    async def replay() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    async def capture(event: Message) -> None:
        events.append(event)

    await app(scope, replay, capture)
    response = json.loads(b"".join(event.get("body", b"") for event in events))
    if "result" in response:
        response["result"]["resultType"] = "complete"
        response["result"].setdefault("_meta", {}).update(result_meta)
    status = next(event["status"] for event in events if event["type"] == "http.response.start")
    return JSONResponse(response, status_code=status, headers={"Cache-Control": "no-store"})
