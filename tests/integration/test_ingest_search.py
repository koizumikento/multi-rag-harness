"""End-to-end ingestion → hybrid search tests with fake models."""

from pathlib import Path

from multi_rag_harness.storage.interfaces import SearchFilters
from tests.conftest import Harness


def write_corpus(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "kuzu.md").write_text(
        "# Kuzu Notes\n\n"
        "## Overview\n\nKuzu is the embedded graph database used for GraphRAG.\n\n"
        "## Errors\n\nBinder exception means the Cypher query referenced a bad column.\n",
        encoding="utf-8",
    )
    (docs / "qdrant.md").write_text(
        "# Qdrant Notes\n\nQdrant runs in embedded local mode for vector search.\n",
        encoding="utf-8",
    )
    (root / "search.py").write_text(
        "def hybrid_search(query):\n    return rrf_merge(keyword(query), vector(query))\n",
        encoding="utf-8",
    )
    skipped = root / "node_modules"
    skipped.mkdir()
    (skipped / "ignored.md").write_text("# Ignored\n\nshould not be ingested\n", encoding="utf-8")


async def test_ingest_and_search(harness: Harness, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)

    report = await harness.pipeline.ingest_path(corpus, tags=["notes"])
    assert report.documents_ingested == 3  # node_modules skipped
    assert report.documents_skipped == 0
    assert report.chunks_indexed >= 4
    assert report.extraction_runs_created == 0  # auto_extract defaults off

    output = await harness.search.search("Binder exception Cypher", rerank=False)
    assert output.reranked is False
    top = output.results[0]
    assert "Binder exception" in top.excerpt
    assert top.kind == "doc"
    assert top.title == "Kuzu Notes > Errors"  # doc title + heading path, deduplicated
    assert top.source_id
    assert top.source_path.endswith("kuzu.md")
    assert top.metadata["rrf_score"] > 0


async def test_code_files_get_kind_and_language(harness: Harness, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    await harness.pipeline.ingest_path(corpus)

    output = await harness.search.search(
        "hybrid_search rrf_merge", filters=SearchFilters(kinds=["code"]), rerank=False
    )
    assert output.results, "code chunk should be found"
    result = output.results[0]
    assert result.kind == "code"
    assert result.source_path.endswith("search.py")

    # Language filter also works.
    output = await harness.search.search(
        "hybrid_search", filters=SearchFilters(language="python"), rerank=False
    )
    assert output.results


async def test_reingest_unchanged_is_skipped(harness: Harness, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    await harness.pipeline.ingest_path(corpus)
    report = await harness.pipeline.ingest_path(corpus)
    assert report.documents_ingested == 0
    assert report.documents_updated == 0
    assert report.documents_skipped == 3

    # No duplicate results for the same chunk.
    output = await harness.search.search("Qdrant embedded local", rerank=False)
    ids = [r.id for r in output.results]
    assert len(ids) == len(set(ids))


async def test_modified_file_replaces_old_chunks(harness: Harness, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    await harness.pipeline.ingest_path(corpus)

    (corpus / "docs" / "qdrant.md").write_text(
        "# Qdrant Notes\n\nQdrant was replaced by pgvector in this scenario.\n",
        encoding="utf-8",
    )
    report = await harness.pipeline.ingest_path(corpus)
    assert report.documents_updated == 1
    assert report.documents_skipped == 2

    stale = await harness.search.search("embedded local mode vector", rerank=False)
    assert all("embedded local mode" not in r.excerpt for r in stale.results)
    fresh = await harness.search.search("replaced by pgvector", rerank=False)
    assert any("replaced by pgvector" in r.excerpt for r in fresh.results)

    # Document identity is preserved on update.
    doc = await harness.storage.metadata.find_document_by_uri(
        str((corpus / "docs" / "qdrant.md").resolve()), "default"
    )
    assert doc is not None
    chunks = await harness.storage.metadata.get_chunks_for_document(doc.id)
    assert len(chunks) >= 1


async def test_rerank_uses_reranker(harness: Harness, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    await harness.pipeline.ingest_path(corpus)

    output = await harness.search.search("graph database GraphRAG", rerank=True)
    assert output.reranked is True
    assert harness.reranker.calls, "reranker should have been invoked"
    assert output.results[0].score > 0


async def test_extraction_runs_created_when_requested(harness: Harness, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    report = await harness.pipeline.ingest_path(corpus, extract=True)
    # Only kind="doc" chunks queue extraction by default.
    assert report.extraction_runs_created > 0
    runs = await harness.storage.metadata.claim_pending_extraction_runs(100)
    assert len(runs) == report.extraction_runs_created


async def test_e5_style_fake_embedder_receives_raw_texts(harness: Harness, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    await harness.pipeline.ingest_path(corpus)
    assert harness.embedder.passage_texts, "passages embedded during ingest"
    await harness.search.search("anything at all", rerank=False)
    assert harness.embedder.query_texts == ["anything at all"]
