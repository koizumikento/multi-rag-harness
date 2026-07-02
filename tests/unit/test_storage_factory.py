"""Unit tests for storage backend factory selection."""

from multi_rag_harness.config import Settings
from multi_rag_harness.storage import build_storage
from multi_rag_harness.storage.pgvector import PgvectorIndex
from multi_rag_harness.storage.postgres import PostgresStore
from multi_rag_harness.storage.qdrant import QdrantVectorIndex
from multi_rag_harness.storage.sqlite import SqliteStore


def test_build_storage_uses_default_local_backends(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data")

    bundle = build_storage(settings)

    assert isinstance(bundle.metadata, SqliteStore)
    assert bundle.keyword is bundle.metadata
    assert isinstance(bundle.vector, QdrantVectorIndex)


def test_build_storage_selects_placeholder_postgres_backends(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.storage.metadata_backend = "postgres"
    settings.storage.vector_backend = "pgvector"
    settings.storage.postgres_dsn = "postgresql://example/db"

    bundle = build_storage(settings)

    assert isinstance(bundle.metadata, PostgresStore)
    assert bundle.keyword is bundle.metadata
    assert isinstance(bundle.vector, PgvectorIndex)
