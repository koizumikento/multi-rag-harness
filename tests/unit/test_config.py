"""Unit tests for configuration loading."""

from pathlib import Path

from multi_rag_harness.config import Settings, load_settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.embedding.model == "intfloat/multilingual-e5-base"
    assert settings.embedding.dimension == 768
    assert settings.reranker.model == "hotchpotch/japanese-reranker-cross-encoder-small-v1"
    assert settings.reranker.enabled_default is True
    assert settings.storage.metadata_backend == "sqlite"
    assert settings.storage.vector_backend == "qdrant"
    assert settings.storage.graph_backend == "kuzu"
    assert settings.mcp.server_name == "multi-rag-harness"
    assert settings.codex.auto_extract_on_ingest is False
    assert settings.codex.extract_kinds == ["doc"]
    assert settings.data_dir == Path("./.multi-rag-harness")


def test_derived_paths() -> None:
    settings = Settings(data_dir=Path("/tmp/mrh-data"))
    assert settings.sqlite_path == Path("/tmp/mrh-data/metadata.db")
    assert settings.qdrant_path == Path("/tmp/mrh-data/qdrant")
    assert settings.kuzu_path == Path("/tmp/mrh-data/kuzu")


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MRH_EMBEDDING__MODEL", "intfloat/multilingual-e5-small")
    monkeypatch.setenv("MRH_EMBEDDING__DIMENSION", "384")
    monkeypatch.setenv("MRH_RERANKER__ENABLED_DEFAULT", "false")
    monkeypatch.setenv("MRH_DATA_DIR", "/tmp/elsewhere")
    settings = load_settings()
    assert settings.embedding.model == "intfloat/multilingual-e5-small"
    assert settings.embedding.dimension == 384
    assert settings.reranker.enabled_default is False
    assert settings.data_dir == Path("/tmp/elsewhere")


def test_toml_file(tmp_path: Path) -> None:
    config_file = tmp_path / "mrh.toml"
    config_file.write_text(
        """
data_dir = "/tmp/from-toml"

[embedding]
model = "BAAI/bge-m3"
dimension = 1024

[reranker]
enabled_default = false

[storage]
qdrant_collection = "custom_vectors"

[mcp]
server_name = "custom-server"
""",
        encoding="utf-8",
    )
    settings = load_settings(config_file)
    assert settings.data_dir == Path("/tmp/from-toml")
    assert settings.embedding.model == "BAAI/bge-m3"
    assert settings.embedding.dimension == 1024
    assert settings.reranker.enabled_default is False
    assert settings.storage.qdrant_collection == "custom_vectors"
    assert settings.mcp.server_name == "custom-server"


def test_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "mrh.toml"
    config_file.write_text('[embedding]\nmodel = "from-toml"\n', encoding="utf-8")
    monkeypatch.setenv("MRH_EMBEDDING__MODEL", "from-env")
    settings = load_settings(config_file)
    assert settings.embedding.model == "from-env"


def test_config_file_env_var(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "pointed.toml"
    config_file.write_text('[mcp]\nserver_name = "pointed"\n', encoding="utf-8")
    monkeypatch.setenv("MRH_CONFIG_FILE", str(config_file))
    settings = load_settings()
    assert settings.mcp.server_name == "pointed"


def test_missing_explicit_config_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    try:
        load_settings(missing)
    except FileNotFoundError as exc:
        assert "nope.toml" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
