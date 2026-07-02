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
    for command in ("serve", "ingest", "extract", "config-show"):
        assert command in result.output
