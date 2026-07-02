"""Shared fixtures: settings, initialized storage, and a fake-model harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from multi_rag_harness.config import EmbeddingSettings, Settings
from multi_rag_harness.ingestion.pipeline import IngestionPipeline
from multi_rag_harness.retrieval.hybrid import HybridRetriever, SearchPipeline
from multi_rag_harness.retrieval.keyword import KeywordRetriever
from multi_rag_harness.retrieval.rerank import RerankService
from multi_rag_harness.retrieval.vector import VectorRetriever
from multi_rag_harness.storage import StorageBundle, build_storage
from tests.fakes import FakeEmbedder, FakeReranker

FAKE_DIMENSION = 32


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        embedding=EmbeddingSettings(dimension=FAKE_DIMENSION),
    )


@dataclass
class Harness:
    settings: Settings
    storage: StorageBundle
    embedder: FakeEmbedder
    reranker: FakeReranker
    pipeline: IngestionPipeline
    search: SearchPipeline
    vector_retriever: VectorRetriever


@pytest.fixture
async def harness(settings: Settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    storage = build_storage(settings)
    await storage.metadata.initialize()
    await storage.vector.initialize(FAKE_DIMENSION)
    await storage.graph.initialize()

    embedder = FakeEmbedder(FAKE_DIMENSION)
    reranker = FakeReranker()
    pipeline = IngestionPipeline(
        storage.metadata, storage.keyword, storage.vector, embedder, settings
    )
    vector_retriever = VectorRetriever(storage.vector, embedder)
    search = SearchPipeline(
        HybridRetriever(KeywordRetriever(storage.keyword), vector_retriever, storage.metadata),
        RerankService(reranker),
        storage.metadata,
        settings,
    )
    yield Harness(
        settings=settings,
        storage=storage,
        embedder=embedder,
        reranker=reranker,
        pipeline=pipeline,
        search=search,
        vector_retriever=vector_retriever,
    )
    await storage.metadata.close()
    await storage.vector.close()
    await storage.graph.close()
