"""Tests for optional Codex SDK client behavior."""

from types import SimpleNamespace

import pytest

from multi_rag_harness.codex import client as codex_client
from multi_rag_harness.codex.client import CodexRunError, CodexSdkClient


async def test_codex_sdk_client_reports_missing_optional_extra(monkeypatch) -> None:
    def fake_import_module(name: str):
        if name == "openai_codex":
            raise ModuleNotFoundError(name)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(codex_client, "import_module", fake_import_module)
    client = CodexSdkClient()

    with pytest.raises(CodexRunError, match="optional 'codex' extra"):
        await client.run_structured("{}", {})


async def test_codex_sdk_client_runs_thread_and_closes(monkeypatch) -> None:
    created = []

    class FakeThread:
        id = "thread-1"

        def __init__(self) -> None:
            self.runs: list[tuple[str, dict]] = []

        async def run(self, prompt: str, output_schema: dict):
            self.runs.append((prompt, output_schema))
            return SimpleNamespace(final_response='{"ok": true}', status="completed")

    class FakeAsyncCodex:
        def __init__(self) -> None:
            self.entered = False
            self.exited = False
            self.thread_kwargs: list[dict] = []
            self.thread = FakeThread()
            created.append(self)

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self.exited = True

        async def thread_start(self, **kwargs):
            self.thread_kwargs.append(kwargs)
            return self.thread

    monkeypatch.setattr(
        codex_client,
        "import_module",
        lambda name: SimpleNamespace(AsyncCodex=FakeAsyncCodex),
    )
    client = CodexSdkClient(model="gpt-test")

    response, thread_id = await client.run_structured("prompt", {"type": "object"})
    assert response == '{"ok": true}'
    assert thread_id == "thread-1"
    assert created[0].entered is True
    assert created[0].thread_kwargs == [{"model": "gpt-test"}]
    assert created[0].thread.runs == [("prompt", {"type": "object"})]

    await client.close()
    assert created[0].exited is True


async def test_codex_sdk_client_wraps_sdk_failures(monkeypatch) -> None:
    class FailingThread:
        id = "thread-err"

        async def run(self, prompt: str, output_schema: dict):
            raise RuntimeError("boom")

    class FakeAsyncCodex:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def thread_start(self, **kwargs):
            return FailingThread()

    monkeypatch.setattr(
        codex_client,
        "import_module",
        lambda name: SimpleNamespace(AsyncCodex=FakeAsyncCodex),
    )
    client = CodexSdkClient()

    with pytest.raises(CodexRunError, match="codex run failed: boom"):
        await client.run_structured("prompt", {})
    await client.close()


async def test_codex_sdk_client_rejects_empty_final_response(monkeypatch) -> None:
    class EmptyThread:
        id = "thread-empty"

        async def run(self, prompt: str, output_schema: dict):
            return SimpleNamespace(final_response=None, status="completed")

    class FakeAsyncCodex:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def thread_start(self, **kwargs):
            return EmptyThread()

    monkeypatch.setattr(
        codex_client,
        "import_module",
        lambda name: SimpleNamespace(AsyncCodex=FakeAsyncCodex),
    )
    client = CodexSdkClient()

    with pytest.raises(CodexRunError, match="no final response: completed"):
        await client.run_structured("prompt", {})
    await client.close()
