"""Authenticated, single-tenant HTTP transport; Sites owns user authentication."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.transport_security import TransportSecurityMiddleware, TransportSecuritySettings
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send

from multi_rag_harness.config import Settings
from multi_rag_harness.mcp_server.modern_http import modern_response
from multi_rag_harness.mcp_server.server import (
    AppContainer,
    build_container,
    close_container,
    create_server,
)

MAX_REQUEST_BYTES = 1024 * 1024


class HttpSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRH_HTTP_", hide_input_in_errors=True)

    backend_token: SecretStr
    site_user_id: str = Field(min_length=1, max_length=256)
    ingest_root: Path
    allowed_host: str = Field(min_length=1)

    @field_validator("backend_token")
    @classmethod
    def strong_token(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("backend token must contain at least 32 characters")
        return value

    @field_validator("ingest_root")
    @classmethod
    def corpus_directory(cls, value: Path) -> Path:
        if not value.is_absolute() or not value.is_dir():
            raise ValueError("ingest root must be an existing absolute corpus directory")
        return value.resolve()


class BackendGateway:
    def __init__(self, http: HttpSettings) -> None:
        self.http = http
        self.app: ASGIApp | None = None
        self.server_info: dict = {}
        self.instructions = ""
        self.security = TransportSecurityMiddleware(
            TransportSecuritySettings(allowed_hosts=[http.allowed_host], allowed_origins=[])
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        status, error = 0, ""
        if request.url.path != "/mcp":
            status, error = 404, "Not found"
        elif not hmac.compare_digest(
            request.headers.get("authorization", "").encode(),
            f"Bearer {self.http.backend_token.get_secret_value()}".encode(),
        ):
            status, error = 401, "Backend authentication required"
        elif request.headers.get("x-mrh-site-user-id") != self.http.site_user_id:
            status, error = 403, "Site user is not authorized for this tenant"
        elif request.method != "POST":
            status, error = 405, "Use POST /mcp"
        if status:
            await JSONResponse(
                {"error": error},
                status_code=status,
                headers={
                    "Cache-Control": "no-store",
                    **({"Allow": "POST"} if status == 405 else {}),
                },
            )(scope, receive, send)
            return

        # Bound chunked bodies too; the SDK validates JSON-RPC and tool schemas.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_REQUEST_BYTES:
                await JSONResponse({"error": "Request too large"}, status_code=413)(
                    scope, receive, send
                )
                return

        invalid = await self.security.validate_request(request, is_post=True)
        if invalid is not None:
            await invalid(scope, receive, send)
            return

        async def replay() -> dict:
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        assert self.app is not None
        response = await modern_response(
            request, bytes(body), self.app, self.server_info, self.instructions
        )
        if response is not None:
            await response(scope, receive, send)
            return
        await self.app(scope, replay, send)


def create_http_app(
    settings: Settings, http: HttpSettings, *, container: AppContainer | None = None
) -> Starlette:
    """One owner, one isolated data_dir/vector collection/backend per Site.

    All object IDs, graph traversal and writes are authorized by the same tenant
    boundary. Logical retrieval scopes are filters, never security boundaries.
    """
    if settings.data_dir.resolve().is_relative_to(http.ingest_root):
        raise ValueError("data_dir must be outside the published corpus root")
    gateway = BackendGateway(http)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        built = container or await build_container(settings)
        try:
            mcp = create_server(settings, container=built, ingest_root=http.ingest_root)
            mcp.settings.stateless_http = True
            mcp.settings.json_response = True
            mcp.settings.transport_security = TransportSecuritySettings(
                allowed_hosts=[http.allowed_host], allowed_origins=[]
            )
            gateway.app = mcp.streamable_http_app()
            options = mcp._mcp_server.create_initialization_options()
            gateway.server_info = {"name": options.server_name, "version": options.server_version}
            gateway.instructions = options.instructions or ""
            async with mcp.session_manager.run():
                yield
        finally:
            if container is None:
                await close_container(built)

    return Starlette(routes=[Mount("/", app=gateway)], lifespan=lifespan)
