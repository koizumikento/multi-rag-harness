"""Configuration models and loading for multi-rag-harness."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_FILE = Path("multi-rag-harness.toml")
CONFIG_FILE_ENV_VAR = "MRH_CONFIG_FILE"


class EmbeddingSettings(BaseModel):
    model: str = "intfloat/multilingual-e5-base"
    device: str = "auto"
    dimension: int = 768
    batch_size: int = 32


class RerankerSettings(BaseModel):
    model: str = "hotchpotch/japanese-reranker-cross-encoder-small-v1"
    enabled_default: bool = True
    device: str = "auto"
    max_candidates: int = 50


class StorageSettings(BaseModel):
    metadata_backend: Literal["sqlite", "postgres"] = "sqlite"
    vector_backend: Literal["qdrant", "pgvector"] = "qdrant"
    graph_backend: Literal["kuzu"] = "kuzu"
    postgres_dsn: str | None = None
    qdrant_collection: str = "mrh_vectors"


class McpSettings(BaseModel):
    server_name: str = "multi-rag-harness"
    instructions: str = (
        "Local-first retrieval and memory substrate. Use rag_search for hybrid "
        "document search, graph_* tools for entity/claim exploration, "
        "trace/decision/error/code/tool_search for typed memory, and "
        "memory_store_* to persist durable memory."
    )


class CodexSettings(BaseModel):
    prompt_version: str = "extraction/v1"
    auto_extract_on_ingest: bool = False
    extract_kinds: list[str] = Field(default_factory=lambda: ["doc"])
    max_runs_per_batch: int = 25


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MRH_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data_dir: Path = Path("./.multi-rag-harness")
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    codex: CodexSettings = Field(default_factory=CodexSettings)

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "metadata.db"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def kuzu_path(self) -> Path:
        return self.data_dir / "kuzu"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit init args > env vars > TOML file > defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def _resolve_config_file(config_file: Path | None) -> Path | None:
    if config_file is not None:
        return config_file
    env_value = os.environ.get(CONFIG_FILE_ENV_VAR)
    if env_value:
        return Path(env_value)
    if DEFAULT_CONFIG_FILE.exists():
        return DEFAULT_CONFIG_FILE
    return None


def load_settings(config_file: Path | None = None) -> Settings:
    """Load settings from an optional TOML file, environment, and defaults.

    Resolution order for the TOML file: explicit argument, then
    ``$MRH_CONFIG_FILE``, then ``./multi-rag-harness.toml`` if present.
    Environment variables (``MRH_`` prefix, ``__`` nesting) override TOML.
    """
    resolved = _resolve_config_file(config_file)
    if resolved is None:
        return Settings()
    if not resolved.exists():
        raise FileNotFoundError(f"config file not found: {resolved}")

    class _TomlSettings(Settings):
        model_config = SettingsConfigDict(
            env_prefix="MRH_",
            env_nested_delimiter="__",
            extra="ignore",
            toml_file=resolved,
        )

    return _TomlSettings()
