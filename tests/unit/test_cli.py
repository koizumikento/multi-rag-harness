"""CLI smoke tests (no model downloads: model factories are patched)."""

from pathlib import Path

from typer.testing import CliRunner

import multi_rag_harness.mcp_server.server as server_module
from multi_rag_harness.cli import app
from tests.fakes import FakeEmbedder, FakeReranker

runner = CliRunner()


def test_config_show(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MRH_DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0
    assert "intfloat/multilingual-e5-base" in result.output
    assert "embedding.model" in result.output


def test_config_show_redacts_api_keys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MRH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MRH_EMBEDDING__API_KEY", "sk-embed")
    monkeypatch.setenv("MRH_RERANKER__API_KEY", "rerank-secret")
    monkeypatch.setenv("MRH_STORAGE__QDRANT_API_KEY", "qdrant-secret")

    result = runner.invoke(app, ["config-show"])

    assert result.exit_code == 0
    assert "embedding.api_key" in result.output
    assert "reranker.api_key" in result.output
    assert "storage.qdrant_api_key" in result.output
    assert "***" in result.output
    assert "sk-embed" not in result.output
    assert "rerank-secret" not in result.output
    assert "qdrant-secret" not in result.output


def test_config_show_with_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "mrh.toml"
    config_file.write_text('[mcp]\nserver_name = "from-toml-server"\n', encoding="utf-8")
    result = runner.invoke(app, ["config-show", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "from-toml-server" in result.output


def test_ingest_command(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MRH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(server_module, "create_embedder", lambda _s: FakeEmbedder(32))
    monkeypatch.setattr(server_module, "create_reranker", lambda _s: FakeReranker())

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("# Note\n\nhello ingestion\n", encoding="utf-8")

    result = runner.invoke(app, ["ingest", str(corpus), "--tag", "cli"])
    assert result.exit_code == 0, result.output
    assert "documents_ingested" in result.output


def test_extract_command_with_no_pending_runs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MRH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(server_module, "create_embedder", lambda _s: FakeEmbedder(32))
    monkeypatch.setattr(server_module, "create_reranker", lambda _s: FakeReranker())

    result = runner.invoke(app, ["extract"])
    assert result.exit_code == 0, result.output
    assert "runs_attempted" in result.output


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("serve", "serve-http", "ingest", "extract", "config-show"):
        assert command in result.output


def test_http_command_uses_env_and_loopback(monkeypatch, tmp_path: Path) -> None:
    import uvicorn

    monkeypatch.setenv("MRH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MRH_HTTP_BACKEND_TOKEN", "fixture-token-with-at-least-32-characters")
    monkeypatch.setenv("MRH_HTTP_SITE_USER_ID", "fixture-owner")
    monkeypatch.setenv("MRH_HTTP_INGEST_ROOT", str(tmp_path))
    monkeypatch.setenv("MRH_HTTP_ALLOWED_HOST", "fixture.example")
    # Storage must sit outside the corpus; no server/process or real model is started.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    monkeypatch.setenv("MRH_HTTP_INGEST_ROOT", str(corpus))
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    result = runner.invoke(app, ["serve-http", "--port", "8765"])
    assert result.exit_code == 0, result.output
    assert calls == [{"host": "127.0.0.1", "port": 8765, "access_log": False}]


def test_http_command_config_error_does_not_print_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MRH_HTTP_BACKEND_TOKEN", "private-short-token")
    monkeypatch.setenv("MRH_HTTP_SITE_USER_ID", "fixture-owner")
    monkeypatch.setenv("MRH_HTTP_INGEST_ROOT", str(tmp_path))
    monkeypatch.setenv("MRH_HTTP_ALLOWED_HOST", "fixture.example")
    result = runner.invoke(app, ["serve-http"])
    assert result.exit_code != 0
    assert "private-short-token" not in result.output
    assert "private-short-token" not in str(result.exception)
    assert not app.pretty_exceptions_show_locals
