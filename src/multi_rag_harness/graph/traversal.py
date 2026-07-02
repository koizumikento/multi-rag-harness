"""Graph search, indexing, and neighborhood traversal services."""

from __future__ import annotations

from collections.abc import Sequence

from multi_rag_harness.graph.models import GraphNeighborhood, GraphSourceRef
from multi_rag_harness.models.embedding import EmbeddingModel
from multi_rag_harness.retrieval.hybrid import rrf_merge
from multi_rag_harness.retrieval.results import SearchResult, make_excerpt
from multi_rag_harness.retrieval.vector import VectorRetriever
from multi_rag_harness.storage.interfaces import (
    ClaimRecord,
    EntityNode,
    GraphStore,
    KeywordIndex,
    MetadataStore,
    ProvenanceRecord,
    SearchFilters,
    VectorIndex,
    VectorPoint,
    utcnow,
)
from multi_rag_harness.storage.qdrant import build_point_payload

MAX_EXPAND_DEPTH = 3
MAX_EXPAND_ENTITIES = 25
MAX_EXPAND_RELATIONS = 50
MAX_EXPAND_CLAIMS = 20
GRAPH_FETCH_N = 50


class GraphIndexer:
    """Indexes entities and claims into the keyword and vector indexes so
    graph items are findable via text search (``item_kind`` separates them
    from document chunks)."""

    def __init__(
        self, keyword: KeywordIndex, vector: VectorIndex, embedder: EmbeddingModel
    ) -> None:
        self._keyword = keyword
        self._vector = vector
        self._embedder = embedder

    async def index_entity(self, entity: EntityNode, aliases: Sequence[str]) -> None:
        parts = [entity.canonical_name, *aliases]
        if entity.description:
            parts.append(entity.description)
        text = " ".join(dict.fromkeys(parts))
        await self._keyword.index_graph_item(entity.id, "entity", text)
        vectors = await self._embedder.embed_passages([text])
        await self._vector.upsert(
            [
                VectorPoint(
                    id=entity.id,
                    vector=vectors[0],
                    payload=build_point_payload(
                        item_id=entity.id,
                        item_kind="entity",
                        kind="entity",
                        source_type="graph",
                        created_at=entity.created_at,
                    ),
                )
            ]
        )

    async def index_claim(self, claim: ClaimRecord, subject_name: str) -> None:
        text = f"{subject_name} {claim.predicate} {claim.object_text}"
        await self._keyword.index_graph_item(claim.id, "claim", text)
        vectors = await self._embedder.embed_passages([text])
        await self._vector.upsert(
            [
                VectorPoint(
                    id=claim.id,
                    vector=vectors[0],
                    payload=build_point_payload(
                        item_id=claim.id,
                        item_kind="claim",
                        kind="claim",
                        source_type="graph",
                        created_at=claim.created_at,
                        valid_from=claim.valid_from,
                        valid_to=claim.valid_to,
                    ),
                )
            ]
        )


class GraphSearchService:
    def __init__(
        self,
        graph: GraphStore,
        metadata: MetadataStore,
        keyword: KeywordIndex,
        vector_retriever: VectorRetriever,
    ) -> None:
        self._graph = graph
        self._metadata = metadata
        self._keyword = keyword
        self._vector = vector_retriever

    async def _hybrid_graph_search(
        self, query: str, item_type: str, filters: SearchFilters | None
    ) -> list[tuple[str, float]]:
        keyword_results = await self._keyword.search_graph_items(query, item_type, GRAPH_FETCH_N)
        vector_results = await self._vector.search(
            query, filters, GRAPH_FETCH_N, item_kinds=[item_type]
        )
        merged = rrf_merge([keyword_results, vector_results])
        return [(s.id, s.score) for s in merged]

    async def _source_for_item(self, item_id: str) -> tuple[str, str]:
        """Return (source_id, source_path) from the first provenance row."""
        rows = await self._metadata.get_provenance_for_item(item_id)
        if not rows:
            return "", ""
        row = rows[0]
        document = await self._metadata.get_document(row.source_id)
        return row.source_id, document.source_uri if document else ""

    async def search_entities(
        self, query: str, filters: SearchFilters | None = None, top_k: int = 10
    ) -> list[SearchResult]:
        merged = await self._hybrid_graph_search(query, "entity", filters)
        results: list[SearchResult] = []
        for entity_id, score in merged:
            if len(results) >= top_k:
                break
            entity = await self._graph.get_entity(entity_id)
            if entity is None:
                continue
            source_id, source_path = await self._source_for_item(entity_id)
            results.append(
                SearchResult(
                    id=entity.id,
                    kind="entity",
                    score=score,
                    source_id=source_id,
                    source_path=source_path,
                    title=entity.canonical_name,
                    excerpt=make_excerpt(entity.description or entity.canonical_name),
                    metadata={"entity_type": entity.entity_type},
                )
            )
        return results

    async def entity_results(
        self, entity_ids: Sequence[str], score: float = 0.0
    ) -> list[SearchResult]:
        """Shape entities as source-grounded search results (used by
        ``rag_search``'s graph_expand enrichment)."""
        results: list[SearchResult] = []
        for entity_id in entity_ids:
            entity = await self._graph.get_entity(entity_id)
            if entity is None:
                continue
            source_id, source_path = await self._source_for_item(entity_id)
            results.append(
                SearchResult(
                    id=entity.id,
                    kind="entity",
                    score=score,
                    source_id=source_id,
                    source_path=source_path,
                    title=entity.canonical_name,
                    excerpt=make_excerpt(entity.description or entity.canonical_name),
                    metadata={"entity_type": entity.entity_type, "graph_expanded": True},
                )
            )
        return results

    async def search_claims(
        self, query: str, filters: SearchFilters | None = None, top_k: int = 10
    ) -> list[SearchResult]:
        merged = await self._hybrid_graph_search(query, "claim", filters)
        confidence_min = filters.confidence_min if filters else None
        claims = await self._metadata.get_claims([claim_id for claim_id, _ in merged])
        claims_by_id = {claim.id: claim for claim in claims}
        results: list[SearchResult] = []
        for claim_id, score in merged:
            if len(results) >= top_k:
                break
            claim = claims_by_id.get(claim_id)
            if claim is None:
                continue
            if confidence_min is not None and claim.confidence < confidence_min:
                continue
            subject = await self._graph.get_entity(claim.subject_entity_id)
            subject_name = subject.canonical_name if subject else claim.subject_entity_id
            source_id, source_path = await self._source_for_item(claim_id)
            statement = f"{subject_name} {claim.predicate} {claim.object_text}"
            results.append(
                SearchResult(
                    id=claim.id,
                    kind="claim",
                    score=score,
                    source_id=source_id,
                    source_path=source_path,
                    title=statement[:120],
                    excerpt=make_excerpt(statement),
                    metadata={
                        "modality": claim.modality,
                        "confidence": claim.confidence,
                        "subject_entity_id": claim.subject_entity_id,
                        "object_entity_id": claim.object_entity_id,
                    },
                )
            )
        return results

    async def expand(
        self,
        entity_id: str,
        depth: int = 1,
        relation_types: Sequence[str] | None = None,
        filters: SearchFilters | None = None,
    ) -> GraphNeighborhood:
        """Breadth-first neighborhood expansion with hard caps."""
        depth = max(1, min(depth, MAX_EXPAND_DEPTH))
        valid_at = filters.valid_at if filters else None
        root = await self._graph.get_entity(entity_id)
        neighborhood = GraphNeighborhood(root_entity_id=entity_id)
        if root is None:
            return neighborhood
        neighborhood.entities.append(root)

        visited = {entity_id}
        seen_relations: set[str] = set()
        frontier = [entity_id]
        for _ in range(depth):
            if not frontier or len(neighborhood.entities) >= MAX_EXPAND_ENTITIES:
                break
            next_frontier: list[str] = []
            for node_id in frontier:
                if len(neighborhood.relations) >= MAX_EXPAND_RELATIONS:
                    break
                for relation, neighbor in await self._graph.get_neighbors(
                    node_id, relation_types, valid_at
                ):
                    if relation.id not in seen_relations:
                        if len(neighborhood.relations) >= MAX_EXPAND_RELATIONS:
                            break
                        seen_relations.add(relation.id)
                        neighborhood.relations.append(relation)
                    if neighbor.id not in visited:
                        if len(neighborhood.entities) < MAX_EXPAND_ENTITIES:
                            visited.add(neighbor.id)
                            neighborhood.entities.append(neighbor)
                            next_frontier.append(neighbor.id)
            frontier = next_frontier

        for entity in neighborhood.entities:
            remaining = MAX_EXPAND_CLAIMS - len(neighborhood.claims)
            if remaining <= 0:
                break
            claims = await self._metadata.get_claims_for_entity(
                entity.id, valid_at or utcnow(), limit=remaining
            )
            existing_ids = {c.id for c in neighborhood.claims}
            neighborhood.claims.extend(c for c in claims if c.id not in existing_ids)
        return neighborhood

    async def get_sources(self, graph_item_id: str) -> list[GraphSourceRef]:
        rows = await self._metadata.get_provenance_for_item(graph_item_id)
        refs: list[GraphSourceRef] = []
        for row in rows:
            refs.append(await self._source_ref(row))
        return refs

    async def _source_ref(self, row: ProvenanceRecord) -> GraphSourceRef:
        document = await self._metadata.get_document(row.source_id)
        excerpt = ""
        if row.chunk_id:
            chunk = await self._metadata.get_chunk(row.chunk_id)
            if chunk is not None:
                excerpt = make_excerpt(chunk.text)
        return GraphSourceRef(
            source_id=row.source_id,
            source_path=document.source_uri if document else "",
            chunk_id=row.chunk_id,
            evidence_text=row.evidence_text,
            excerpt=excerpt,
        )
