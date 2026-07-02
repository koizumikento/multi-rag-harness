"""Integration tests for the Codex extraction flow with a fake client."""

import json
from pathlib import Path

import pytest

from multi_rag_harness.codex.extraction import ExtractionOrchestrator
from multi_rag_harness.graph.canonicalize import EntityCanonicalizer
from multi_rag_harness.graph.traversal import GraphIndexer, GraphSearchService
from tests.conftest import Harness
from tests.fakes import FakeCodexClient

DOC_TEXT = (
    "# Graph Notes\n\n"
    "Kuzu is the embedded graph store. Kuzu powers GraphRAG traversal in this harness.\n"
)

VALID_RESPONSE = json.dumps(
    {
        "entities": [
            {
                "name": "Kuzu",
                "type": "technology",
                "aliases": ["KuzuDB"],
                "description": "embedded graph store",
                "confidence": 0.9,
                "evidence": "Kuzu is the embedded graph store",
            },
            {
                "name": "GraphRAG",
                "type": "concept",
                "aliases": [],
                "description": "graph-based retrieval",
                "confidence": 0.85,
                "evidence": "Kuzu powers GraphRAG traversal",
            },
        ],
        "relations": [
            {
                "source": "Kuzu",
                "target": "GraphRAG",
                "type": "powers",
                "description": "",
                "confidence": 0.8,
                "evidence": "Kuzu powers GraphRAG traversal",
                "valid_from": None,
                "valid_to": None,
            },
            {
                "source": "Kuzu",
                "target": "Unknown Thing",
                "type": "relates_to",
                "description": "",
                "confidence": 0.5,
                "evidence": "Kuzu powers GraphRAG traversal",
                "valid_from": None,
                "valid_to": None,
            },
        ],
        "claims": [
            {
                "subject": "Kuzu",
                "predicate": "is",
                "object": "embedded graph store",
                "modality": "fact",
                "confidence": 0.9,
                "evidence": "Kuzu is the embedded graph store",
                "valid_from": None,
                "valid_to": None,
            }
        ],
    }
)


@pytest.fixture
def graph_search(harness: Harness) -> GraphSearchService:
    return GraphSearchService(
        harness.storage.graph,
        harness.storage.metadata,
        harness.storage.keyword,
        harness.vector_retriever,
    )


def make_orchestrator(harness: Harness, codex: FakeCodexClient) -> ExtractionOrchestrator:
    return ExtractionOrchestrator(
        codex,
        harness.storage.metadata,
        harness.storage.graph,
        EntityCanonicalizer(harness.storage.metadata, harness.storage.graph),
        GraphIndexer(harness.storage.keyword, harness.storage.vector, harness.embedder),
        harness.settings,
    )


async def ingest_docs(harness: Harness, tmp_path: Path, *texts: str) -> None:
    corpus = tmp_path / "extract-corpus"
    corpus.mkdir()
    for index, text in enumerate(texts):
        (corpus / f"doc-{index}.md").write_text(text, encoding="utf-8")
    await harness.pipeline.ingest_path(corpus, extract=True)


async def test_extraction_persists_graph(
    harness: Harness, graph_search: GraphSearchService, tmp_path: Path
) -> None:
    await ingest_docs(harness, tmp_path, DOC_TEXT)
    codex = FakeCodexClient([VALID_RESPONSE])
    orchestrator = make_orchestrator(harness, codex)

    summary = await orchestrator.run_pending()
    assert summary.runs_attempted == 1
    assert summary.runs_completed == 1
    assert summary.runs_failed == 0
    assert summary.entities_created == 2
    assert summary.relations_created == 1  # unresolved endpoint one is skipped
    assert summary.claims_created == 1
    assert "Kuzu is the embedded graph store" in codex.prompts[0]

    # Entities are searchable with provenance-backed sources.
    entities = await graph_search.search_entities("Kuzu")
    assert entities
    top = entities[0]
    assert top.kind == "entity"
    assert top.title == "Kuzu"
    assert top.source_path.endswith("doc-0.md")

    # Alias lookup resolved into the same entity.
    alias_hit = await graph_search.search_entities("KuzuDB")
    assert alias_hit and alias_hit[0].id == top.id

    # Claims are searchable.
    claims = await graph_search.search_claims("embedded graph store")
    assert claims
    assert claims[0].kind == "claim"
    assert claims[0].metadata["modality"] == "fact"

    # Neighborhood expansion reaches GraphRAG via the relation.
    neighborhood = await graph_search.expand(top.id, depth=1)
    names = {entity.canonical_name for entity in neighborhood.entities}
    assert names == {"Kuzu", "GraphRAG"}
    assert len(neighborhood.relations) == 1
    assert neighborhood.relations[0].relation_type == "powers"
    assert any(claim.predicate == "is" for claim in neighborhood.claims)

    # Provenance sources resolve to the ingested document.
    sources = await graph_search.get_sources(top.id)
    assert sources
    assert sources[0].evidence_text == "Kuzu is the embedded graph store"
    assert sources[0].source_path.endswith("doc-0.md")
    assert sources[0].excerpt


async def test_extraction_failure_marks_run_failed(harness: Harness, tmp_path: Path) -> None:
    await ingest_docs(harness, tmp_path, DOC_TEXT, "# Other\n\nSecond document body.\n")
    codex = FakeCodexClient([VALID_RESPONSE, "this is not json"])
    orchestrator = make_orchestrator(harness, codex)

    summary = await orchestrator.run_pending()
    assert summary.runs_attempted == 2
    assert summary.runs_completed == 1
    assert summary.runs_failed == 1

    # No pending runs remain, and nothing is left claimable.
    assert await harness.storage.metadata.claim_pending_extraction_runs(10) == []


async def test_second_run_merges_entities(harness: Harness, tmp_path: Path) -> None:
    await ingest_docs(harness, tmp_path, DOC_TEXT)
    codex = FakeCodexClient([VALID_RESPONSE])
    orchestrator = make_orchestrator(harness, codex)
    await orchestrator.run_pending()

    # Re-ingest an updated document referencing the same entity.
    corpus = tmp_path / "extract-corpus"
    (corpus / "doc-0.md").write_text(DOC_TEXT + "\nKuzu also supports Cypher.\n", encoding="utf-8")
    await harness.pipeline.ingest_path(corpus, extract=True)

    merge_response = json.dumps(
        {
            "entities": [
                {
                    "name": "kuzu",
                    "type": "",
                    "aliases": [],
                    "description": "",
                    "confidence": 0.7,
                    "evidence": "Kuzu also supports Cypher",
                }
            ],
            "relations": [],
            "claims": [],
        }
    )
    codex2 = FakeCodexClient([merge_response, merge_response])
    orchestrator2 = make_orchestrator(harness, codex2)
    summary = await orchestrator2.run_pending()
    assert summary.entities_created == 0
    assert summary.entities_merged >= 1
