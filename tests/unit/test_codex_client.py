"""Tests for optional Codex SDK client behavior."""

import builtins

import pytest

from multi_rag_harness.codex.client import CodexRunError, CodexSdkClient


async def test_codex_sdk_client_reports_missing_optional_extra(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai_codex":
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    client = CodexSdkClient()

    with pytest.raises(CodexRunError, match="optional 'codex' extra"):
        await client.run_structured("{}", {})
