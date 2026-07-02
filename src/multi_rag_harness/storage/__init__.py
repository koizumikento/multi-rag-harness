"""Storage adapter package: backend selection behind neutral interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from multi_rag_harness.config import Settings
from multi_rag_harness.storage.interfaces import (
    GraphStore,
    KeywordIndex,
    MetadataStore,
    VectorIndex,
)
from multi_rag_harness.storage.kuzu import KuzuGraphStore
from multi_rag_harness.storage.pgvector import PgvectorIndex
from multi_rag_harness.storage.postgres import PostgresStore
from multi_rag_harness.storage.qdrant import QdrantVectorIndex
from multi_rag_harness.storage.sqlite import SqliteStore

__all__ = ["StorageBundle", "build_storage"]


@dataclass
class StorageBundle:
    metadata: MetadataStore
    keyword: KeywordIndex
    vector: VectorIndex
    graph: GraphStore


def build_storage(settings: Settings) -> StorageBundle:
    """Construct storage adapters from settings. Construction never touches
    the backends; call ``initialize()`` on each adapter before use."""
    metadata: MetadataStore
    keyword: KeywordIndex
    if settings.storage.metadata_backend == "sqlite":
        sqlite_store = SqliteStore(settings.sqlite_path)
        metadata = sqlite_store
        keyword = sqlite_store
    else:
        # Placeholder adapters; cast because they satisfy the protocols only
        # dynamically (every call raises NotImplementedError).
        postgres_store = PostgresStore(settings.storage.postgres_dsn)
        metadata = cast(MetadataStore, postgres_store)
        keyword = cast(KeywordIndex, postgres_store)

    vector: VectorIndex
    if settings.storage.vector_backend == "qdrant":
        vector = QdrantVectorIndex(
            settings.qdrant_path,
            settings.storage.qdrant_collection,
            url=settings.storage.qdrant_url,
            api_key=settings.storage.qdrant_api_key,
        )
    else:
        vector = cast(VectorIndex, PgvectorIndex(settings.storage.postgres_dsn))

    graph: GraphStore = KuzuGraphStore(settings.kuzu_path)
    return StorageBundle(metadata=metadata, keyword=keyword, vector=vector, graph=graph)
