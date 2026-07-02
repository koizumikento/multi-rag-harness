"""Tests for explicit not-yet-implemented storage adapters."""

import pytest

from multi_rag_harness.storage.pgvector import PgvectorIndex
from multi_rag_harness.storage.postgres import PostgresStore


async def test_postgres_placeholder_methods_raise_clear_error() -> None:
    store = PostgresStore("postgresql://example/db")
    with pytest.raises(NotImplementedError, match="postgres backend not implemented"):
        await store.initialize()
    with pytest.raises(NotImplementedError, match="postgres backend not implemented"):
        await store.search("query")
    await store.close()


async def test_pgvector_placeholder_methods_raise_clear_error() -> None:
    index = PgvectorIndex("postgresql://example/db")
    with pytest.raises(NotImplementedError, match="pgvector backend not implemented"):
        await index.initialize(768)
    with pytest.raises(NotImplementedError, match="pgvector backend not implemented"):
        await index.search([0.0], top_k=1)
    await index.close()
