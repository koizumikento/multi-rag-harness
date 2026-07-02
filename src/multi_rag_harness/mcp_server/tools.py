"""MCP tool implementations: thin handlers delegating to services.

The tool surface is fixed by the specification (15 tools). Handlers pull the
service container from the lifespan context and shape responses; no storage
or retrieval logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context, FastMCP

from multi_rag_harness.mcp_server.schemas import (
    ChunkContext,
    FiltersInput,
    GetSourceResponse,
    GraphExpandResponse,
    GraphSourcesResponse,
    IngestOptions,
    IngestResponse,
    SearchResponse,
    StoreResponse,
    resolve_filters,
)
from multi_rag_harness.memory.decisions import DecisionPayload
from multi_rag_harness.memory.failures import FailurePayload
from multi_rag_harness.memory.traces import TracePayload
from multi_rag_harness.retrieval.results import SearchOutput, SearchResult

if TYPE_CHECKING:
    from multi_rag_harness.mcp_server.server import AppContainer

GRAPH_EXPAND_MAX_ENTITIES = 5


def _container(ctx: Context) -> AppContainer:
    return ctx.request_context.lifespan_context


def _search_response(query: str, output: SearchOutput) -> SearchResponse:
    return SearchResponse(results=output.results, query=query, reranked=output.reranked)


def _graph_search_response(query: str, results: list[SearchResult]) -> SearchResponse:
    return SearchResponse(results=results, query=query, reranked=False)


def register_tools(mcp: FastMCP) -> None:  # noqa: C901
    @mcp.tool()
    async def rag_ingest_path(
        path: str,
        ctx: Context,
        scope: str = "default",
        tags: list[str] | None = None,
        options: IngestOptions | None = None,
    ) -> IngestResponse:
        """Ingest a file or directory into the retrieval indexes. Markdown and
        text become kind="doc"; recognized source files become kind="code"
        with language detection. Re-ingesting unchanged files is a no-op."""
        container = _container(ctx)
        options = options or IngestOptions()
        report = await container.pipeline.ingest_path(
            Path(path),
            scope=scope,
            tags=tags,
            kind_override=options.kind,
            extract=options.extract,
        )
        return IngestResponse(**report.model_dump())

    @mcp.tool()
    async def rag_search(
        query: str,
        ctx: Context,
        scopes: list[str] | None = None,
        kinds: list[str] | None = None,
        filters: FiltersInput | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
        graph_expand: bool = False,
    ) -> SearchResponse:
        """Hybrid search (keyword + vector + RRF) over all indexed content
        with optional local reranking. Use kinds to target doc/code/trace/
        decision/error/tool chunks. graph_expand=true appends up to 5 graph
        entities linked to the returned chunks."""
        container = _container(ctx)
        search_filters = resolve_filters(filters, scopes, kinds)
        output = await container.search.search(query, search_filters, top_k, rerank)
        response = _search_response(query, output)
        if graph_expand and output.results:
            entity_ids = await container.storage.metadata.get_entity_ids_for_chunks(
                [result.id for result in output.results]
            )
            response.results.extend(
                await container.graph_search.entity_results(entity_ids[:GRAPH_EXPAND_MAX_ENTITIES])
            )
        return response

    @mcp.tool()
    async def rag_get_source(
        source_id: str,
        ctx: Context,
        around: int = 1,
    ) -> GetSourceResponse:
        """Fetch full source context for a search result. Accepts a chunk id
        (returns that chunk with ±around neighbors) or a document id (returns
        the first chunks of the document)."""
        container = _container(ctx)
        metadata = container.storage.metadata
        chunk = await metadata.get_chunk(source_id)
        if chunk is not None:
            chunks = await metadata.get_chunk_window(source_id, around)
            document = await metadata.get_document(chunk.document_id)
        else:
            document = await metadata.get_document(source_id)
            if document is None:
                raise ValueError(f"source not found: {source_id}")
            chunks = await metadata.get_chunks_for_document(document.id, limit=2 * around + 1)
        return GetSourceResponse(
            source_id=document.id if document else source_id,
            source_path=document.source_uri if document else "",
            title=document.title if document else "",
            kind=document.kind if document else "doc",
            chunks=[
                ChunkContext(
                    chunk_id=c.id,
                    ordinal=c.ordinal,
                    heading_path=c.heading_path,
                    text=c.text,
                    is_target=c.id == source_id,
                )
                for c in chunks
            ],
            metadata=document.metadata if document else {},
        )

    @mcp.tool()
    async def graph_search_entities(
        query: str,
        ctx: Context,
        filters: FiltersInput | None = None,
        top_k: int = 10,
    ) -> SearchResponse:
        """Search knowledge-graph entities by name, alias, or description.
        Returned ids feed graph_expand and graph_get_sources."""
        container = _container(ctx)
        results = await container.graph_search.search_entities(
            query, resolve_filters(filters), top_k
        )
        return _graph_search_response(query, results)

    @mcp.tool()
    async def graph_search_claims(
        query: str,
        ctx: Context,
        filters: FiltersInput | None = None,
        top_k: int = 10,
    ) -> SearchResponse:
        """Search extracted claims (subject-predicate-object statements with
        modality and confidence). Use filters.confidence_min to cut noise."""
        container = _container(ctx)
        results = await container.graph_search.search_claims(query, resolve_filters(filters), top_k)
        return _graph_search_response(query, results)

    @mcp.tool()
    async def graph_expand(
        entity_id: str,
        ctx: Context,
        depth: int = 1,
        relation_types: list[str] | None = None,
        filters: FiltersInput | None = None,
    ) -> GraphExpandResponse:
        """Expand an entity's neighborhood (BFS, depth 1-3, capped output):
        related entities, relations, and claims about the included entities."""
        container = _container(ctx)
        neighborhood = await container.graph_search.expand(
            entity_id, depth, relation_types, resolve_filters(filters)
        )
        return GraphExpandResponse.from_neighborhood(neighborhood)

    @mcp.tool()
    async def graph_get_sources(
        graph_item_id: str,
        ctx: Context,
    ) -> GraphSourcesResponse:
        """Fetch provenance for a graph item (entity, relation, or claim id):
        source documents, chunks, and the verbatim evidence spans."""
        container = _container(ctx)
        sources = await container.graph_search.get_sources(graph_item_id)
        return GraphSourcesResponse(graph_item_id=graph_item_id, sources=sources)

    @mcp.tool()
    async def trace_search(
        query: str,
        ctx: Context,
        filters: FiltersInput | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchResponse:
        """Search past task execution traces (what was done, which tools and
        commands ran, what failed, the outcome)."""
        container = _container(ctx)
        output = await container.traces.search(query, resolve_filters(filters), top_k, rerank)
        return _search_response(query, output)

    @mcp.tool()
    async def decision_search(
        query: str,
        ctx: Context,
        filters: FiltersInput | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchResponse:
        """Search durable technical decisions and their rationale. Superseded
        decisions are hidden by default (filters.include_expired shows them)."""
        container = _container(ctx)
        output = await container.decisions.search(query, resolve_filters(filters), top_k, rerank)
        return _search_response(query, output)

    @mcp.tool()
    async def error_search(
        query: str,
        ctx: Context,
        filters: FiltersInput | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchResponse:
        """Search known failures by error text, cause, or fix. Best with exact
        error messages as the query."""
        container = _container(ctx)
        output = await container.failures.search(query, resolve_filters(filters), top_k, rerank)
        return _search_response(query, output)

    @mcp.tool()
    async def code_search(
        query: str,
        ctx: Context,
        filters: FiltersInput | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchResponse:
        """Search ingested source code (kind="code" chunks). Supports
        filters.language and filters.path_prefix."""
        container = _container(ctx)
        output = await container.code.search(query, resolve_filters(filters), top_k, rerank)
        return _search_response(query, output)

    @mcp.tool()
    async def tool_search(
        query: str,
        ctx: Context,
        filters: FiltersInput | None = None,
        top_k: int = 10,
        rerank: bool | None = None,
    ) -> SearchResponse:
        """Search stored MCP tool descriptions to pick the right tool for a
        task (descriptions, schemas, examples, known failure modes)."""
        container = _container(ctx)
        output = await container.tools_memory.search(query, resolve_filters(filters), top_k, rerank)
        return _search_response(query, output)

    @mcp.tool()
    async def memory_store_trace(
        payload: TracePayload,
        ctx: Context,
    ) -> StoreResponse:
        """Store a task execution trace as durable, searchable memory."""
        container = _container(ctx)
        ref = await container.traces.store(payload)
        return StoreResponse(record_id=ref.record_id, document_id=ref.document_id)

    @mcp.tool()
    async def memory_store_decision(
        payload: DecisionPayload,
        ctx: Context,
    ) -> StoreResponse:
        """Store a technical decision with rationale. Set payload.supersedes
        to an older decision id to replace it (the old one is hidden from
        default searches)."""
        container = _container(ctx)
        ref = await container.decisions.store(payload)
        return StoreResponse(record_id=ref.record_id, document_id=ref.document_id)

    @mcp.tool()
    async def memory_store_failure(
        payload: FailurePayload,
        ctx: Context,
    ) -> StoreResponse:
        """Store a failure signature with cause and fix so future error_search
        calls can find the resolution."""
        container = _container(ctx)
        ref = await container.failures.store(payload)
        return StoreResponse(record_id=ref.record_id, document_id=ref.document_id)

    _ = (
        rag_ingest_path,
        rag_search,
        rag_get_source,
        graph_search_entities,
        graph_search_claims,
        graph_expand,
        graph_get_sources,
        trace_search,
        decision_search,
        error_search,
        code_search,
        tool_search,
        memory_store_trace,
        memory_store_decision,
        memory_store_failure,
    )
