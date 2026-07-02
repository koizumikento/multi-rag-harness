"""Codex SDK client helpers.

Non-interactive Codex jobs (graph extraction, curation) go through the
``CodexClient`` protocol so orchestration code is testable with fakes and the
SDK stays isolated to this module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol, runtime_checkable


class CodexRunError(RuntimeError):
    """A Codex run failed or returned no final response."""


@runtime_checkable
class CodexClient(Protocol):
    async def run_structured(
        self, prompt: str, output_schema: dict[str, Any]
    ) -> tuple[str, str | None]:
        """Run a non-interactive turn. Returns ``(final_response, thread_id)``.

        Raises :class:`CodexRunError` on failure or empty output.
        """
        ...


class CodexSdkClient:
    """Thin wrapper over the openai-codex SDK. Lazily initialized; each call
    runs in a fresh thread so per-chunk extraction jobs stay isolated."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model
        self._codex: Any | None = None

    async def _client(self) -> Any:
        if self._codex is None:
            try:
                codex_module = import_module("openai_codex")
            except ModuleNotFoundError as exc:
                raise CodexRunError(
                    "Codex SDK extraction requires the optional 'codex' extra. "
                    "Install it with: uv sync --extra codex"
                ) from exc

            AsyncCodex = codex_module.AsyncCodex
            codex = AsyncCodex()
            await codex.__aenter__()
            self._codex = codex
        return self._codex

    async def close(self) -> None:
        if self._codex is not None:
            await self._codex.__aexit__(None, None, None)
            self._codex = None

    async def run_structured(
        self, prompt: str, output_schema: dict[str, Any]
    ) -> tuple[str, str | None]:
        codex = await self._client()
        thread = await codex.thread_start(**({"model": self._model} if self._model else {}))
        try:
            result = await thread.run(prompt, output_schema=output_schema)
        except Exception as exc:  # SDK errors become CodexRunError for callers
            raise CodexRunError(f"codex run failed: {exc}") from exc
        thread_id = getattr(thread, "id", None)
        if result.final_response is None:
            raise CodexRunError(f"codex run returned no final response: {result.status}")
        return result.final_response, thread_id
