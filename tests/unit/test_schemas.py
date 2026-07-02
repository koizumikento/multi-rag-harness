"""Unit tests for MCP tool schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from multi_rag_harness.graph.models import GraphNeighborhood
from multi_rag_harness.mcp_server.schemas import (
    FiltersInput,
    GraphExpandResponse,
    SearchResponse,
    StoreResponse,
    resolve_filters,
)
from multi_rag_harness.memory.decisions import DecisionPayload
from multi_rag_harness.retrieval.results import SearchResult
from multi_rag_harness.storage.interfaces import ClaimRecord, EntityNode, RelationEdge


def test_search_result_has_exact_spec_fields() -> None:
    assert set(SearchResult.model_fields) == {
        "id",
        "kind",
        "score",
        "source_id",
        "source_path",
        "title",
        "excerpt",
        "metadata",
    }


def test_filters_input_maps_to_search_filters() -> None:
    filters = FiltersInput(
        repo="mrh",
        path_prefix="src/",
        language="python",
        tags=["x"],
        source_type="file",
        created_after=datetime(2026, 1, 1, tzinfo=UTC),
        include_expired=True,
        confidence_min=0.5,
    )
    mapped = filters.to_search_filters(scopes=["s1"], kinds=["doc", "code"])
    assert mapped.scopes == ["s1"]
    assert mapped.kinds == ["doc", "code"]
    assert mapped.repo == "mrh"
    assert mapped.path_prefix == "src/"
    assert mapped.language == "python"
    assert mapped.tags == ["x"]
    assert mapped.source_type == "file"
    assert mapped.created_after == datetime(2026, 1, 1, tzinfo=UTC)
    assert mapped.include_expired is True
    assert mapped.confidence_min == 0.5


def test_resolve_filters_defaults() -> None:
    mapped = resolve_filters(None, kinds=["trace"])
    assert mapped.kinds == ["trace"]
    assert mapped.include_expired is False


def test_decision_payload_status_literal() -> None:
    DecisionPayload(title="t", decision="d", status="proposed")
    with pytest.raises(ValidationError):
        DecisionPayload.model_validate({"title": "t", "decision": "d", "status": "maybe"})


def test_graph_expand_response_from_neighborhood() -> None:
    entity_a = EntityNode(canonical_name="A", entity_type="tech")
    entity_b = EntityNode(canonical_name="B")
    relation = RelationEdge(
        source_entity_id=entity_a.id, target_entity_id=entity_b.id, relation_type="uses"
    )
    claim = ClaimRecord(
        subject_entity_id=entity_a.id,
        predicate="is",
        object_text="x",
        modality="fact",
        confidence=0.9,
    )
    neighborhood = GraphNeighborhood(
        root_entity_id=entity_a.id,
        entities=[entity_a, entity_b],
        relations=[relation],
        claims=[claim],
    )
    response = GraphExpandResponse.from_neighborhood(neighborhood)
    assert response.root_entity_id == entity_a.id
    assert [e.canonical_name for e in response.entities] == ["A", "B"]
    assert response.relations[0].relation_type == "uses"
    assert response.claims[0].predicate == "is"
    # Compact summaries: no metadata blobs.
    assert "metadata" not in type(response.entities[0]).model_fields


def test_misc_response_models() -> None:
    StoreResponse(record_id="r", document_id="d")
    SearchResponse(results=[], query="q", reranked=False)
