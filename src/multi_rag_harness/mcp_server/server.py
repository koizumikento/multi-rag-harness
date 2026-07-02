"""MCP server assembly: service container, lifespan, and stdio entry."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from multi_rag_harness.codex.client import CodexClient, CodexSdkClient
from multi_rag_harness.codex.extraction import ExtractionOrchestrator
from multi_rag_harness.config import Settings
from multi_rag_harness.graph.canonicalize import EntityCanonicalizer
from multi_rag_harness.graph.communities import CommunityService
from multi_rag_harness.graph.traversal import GraphIndexer, GraphSearchService
from multi_rag_harness.ingestion.pipeline import IngestionPipeline
from multi_rag_harness.memory.code import CodeSearchService
from multi_rag_harness.memory.decisions import DecisionMemoryService
from multi_rag_harness.memory.failures import FailureMemoryService
from multi_rag_harness.memory.tools import ToolMemoryService
from multi_rag_harness.memory.traces import TraceMemoryService
from multi_rag_harness.models.embedding import EmbeddingModel
from multi_rag_harness.models.local import create_embedder, create_reranker
from multi_rag_harness.models.reranker import Reranker
from multi_rag_harness.retrieval.hybrid import HybridRetriever, SearchPipeline
from multi_rag_harness.retrieval.keyword import KeywordRetriever
from multi_rag_harness.retrieval.rerank import RerankService
from multi_rag_harness.retrieval.vector import VectorRetriever
from multi_rag_harness.storage import StorageBundle, build_storage


@dataclass
class AppContainer:
    settings: Settings
    storage: StorageBundle
    embedder: EmbeddingModel
    reranker: Reranker
    codex_client: CodexClient
    search: SearchPipeline
    pipeline: IngestionPipeline
    graph_search: GraphSearchService
    communities: CommunityService
    traces: TraceMemoryService
    decisions: DecisionMemoryService
    failures: FailureMemoryService
    tools_memory: ToolMemoryService
    code: CodeSearchService
    extraction: ExtractionOrchestrator


async def build_container(
    settings: Settings,
    *,
    embedder: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    codex_client: CodexClient | None = None,
) -> AppContainer:
    """Build and initialize all services. The keyword-only overrides are the
    dependency-injection seam for tests (fake models, fake Codex)."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    storage = build_storage(settings)
    embedder = embedder or create_embedder(settings)
    reranker = reranker or create_reranker(settings)
    codex_client = codex_client or CodexSdkClient(model=settings.codex.model)

    await storage.metadata.initialize()
    await storage.vector.initialize(embedder.dimension)
    await storage.graph.initialize()

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
    graph_search = GraphSearchService(
        storage.graph, storage.metadata, storage.keyword, vector_retriever
    )
    extraction = ExtractionOrchestrator(
        codex_client,
        storage.metadata,
        storage.graph,
        EntityCanonicalizer(storage.metadata, storage.graph),
        GraphIndexer(storage.keyword, storage.vector, embedder),
        settings,
    )
    return AppContainer(
        settings=settings,
        storage=storage,
        embedder=embedder,
        reranker=reranker,
        codex_client=codex_client,
        search=search,
        pipeline=pipeline,
        graph_search=graph_search,
        communities=CommunityService(storage.graph, storage.metadata),
        traces=TraceMemoryService(storage.metadata, pipeline, search),
        decisions=DecisionMemoryService(storage.metadata, pipeline, search),
        failures=FailureMemoryService(storage.metadata, pipeline, search),
        tools_memory=ToolMemoryService(storage.metadata, pipeline, search),
        code=CodeSearchService(search),
        extraction=extraction,
    )


async def close_container(container: AppContainer) -> None:
    if isinstance(container.codex_client, CodexSdkClient):
        await container.codex_client.close()
    # API-backed model adapters hold HTTP clients; local ones have no close.
    for model in (container.embedder, container.reranker):
        closer = getattr(model, "close", None)
        if callable(closer):
            await closer()
    await container.storage.metadata.close()
    await container.storage.vector.close()
    await container.storage.graph.close()


def create_server(
    settings: Settings,
    *,
    embedder: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    codex_client: CodexClient | None = None,
    container: AppContainer | None = None,
) -> FastMCP:
    """Assemble the FastMCP server. A prebuilt ``container`` skips lifecycle
    management (used by tests that seed data through the container)."""
    from multi_rag_harness.mcp_server.tools import register_tools

    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[AppContainer]:
        if container is not None:
            yield container
            return
        built = await build_container(
            settings, embedder=embedder, reranker=reranker, codex_client=codex_client
        )
        try:
            yield built
        finally:
            await close_container(built)

    mcp = FastMCP(
        settings.mcp.server_name,
        instructions=settings.mcp.instructions,
        lifespan=lifespan,
    )
    register_tools(mcp)
    return mcp


def run_stdio(settings: Settings) -> None:
    create_server(settings).run(transport="stdio")
