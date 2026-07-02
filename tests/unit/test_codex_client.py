"""Tests for optional Codex SDK client behavior."""

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
