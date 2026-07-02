"""Ingestion pipeline orchestration.

Persists documents and chunks, feeds the keyword and vector indexes, and
optionally queues graph extraction runs. Memory records reuse the same path
via ``ingest_memory_record`` so every memory kind is searchable through the
one hybrid retrieval contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from multi_rag_harness.config import Settings
from multi_rag_harness.ingestion.chunking import ChunkDraft, chunk_code, chunk_markdown, chunk_text
from multi_rag_harness.ingestion.documents import LoadedDocument, iter_documents
from multi_rag_harness.models.embedding import EmbeddingModel
from multi_rag_harness.storage.interfaces import (
    ChunkRecord,
    DocumentRecord,
    ExtractionRunRecord,
    KeywordIndex,
    MetadataStore,
    VectorIndex,
    VectorPoint,
    new_id,
    utcnow,
)
from multi_rag_harness.storage.qdrant import build_point_payload


class IngestReport(BaseModel):
    documents_ingested: int = 0
    documents_skipped: int = 0
    documents_updated: int = 0
    chunks_indexed: int = 0
    extraction_runs_created: int = 0


def _draft_chunks(loaded: LoadedDocument) -> list[ChunkDraft]:
    if loaded.kind == "code":
        return chunk_code(loaded.text)
    if loaded.source_uri.lower().endswith((".md", ".markdown")):
        return chunk_markdown(loaded.text)
    return chunk_text(loaded.text)


class IngestionPipeline:
    def __init__(
        self,
        metadata: MetadataStore,
        keyword: KeywordIndex,
        vector: VectorIndex,
        embedder: EmbeddingModel,
        settings: Settings,
    ) -> None:
        self._metadata = metadata
        self._keyword = keyword
        self._vector = vector
        self._embedder = embedder
        self._settings = settings

    async def ingest_path(
        self,
        path: Path,
        scope: str = "default",
        tags: list[str] | None = None,
        *,
        kind_override: str | None = None,
        extract: bool | None = None,
    ) -> IngestReport:
        report = IngestReport()
        do_extract = self._settings.codex.auto_extract_on_ingest if extract is None else extract
        for loaded in iter_documents(path):
            kind = kind_override or loaded.kind
            existing = await self._metadata.find_document_by_uri(loaded.source_uri, scope)
            if existing is not None and existing.content_hash == loaded.content_hash:
                report.documents_skipped += 1
                continue

            if existing is not None:
                await self._remove_existing_chunks(existing.id)
                document = existing.model_copy(
                    update={
                        "title": loaded.title,
                        "content_hash": loaded.content_hash,
                        "kind": kind,
                        "language": loaded.language,
                        "tags": list(tags or existing.tags),
                        "updated_at": utcnow(),
                    }
                )
                report.documents_updated += 1
            else:
                document = DocumentRecord(
                    source_type="file",
                    source_uri=loaded.source_uri,
                    title=loaded.title,
                    content_hash=loaded.content_hash,
                    scope=scope,
                    kind=kind,
                    language=loaded.language,
                    tags=list(tags or []),
                )
                report.documents_ingested += 1

            chunks = await self._index_document(document, _draft_chunks(loaded), loaded.path)
            report.chunks_indexed += len(chunks)

            if do_extract and kind in self._settings.codex.extract_kinds:
                runs = [
                    ExtractionRunRecord(
                        source_id=document.id,
                        chunk_id=chunk.id,
                        prompt_version=self._settings.codex.prompt_version,
                    )
                    for chunk in chunks
                ]
                await self._metadata.create_extraction_runs(runs)
                report.extraction_runs_created += len(runs)
        return report

    async def ingest_memory_record(
        self,
        *,
        kind: str,
        record_id: str,
        title: str,
        text: str,
        scope: str = "default",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Index a rendered memory record as a searchable document.

        The typed table row remains authoritative; the document links back via
        ``metadata.record_id``. Returns the document id.
        """
        source_uri = f"memory://{kind}/{record_id}"
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_metadata = dict(metadata or {})
        doc_metadata["record_id"] = record_id
        existing = await self._metadata.find_document_by_uri(source_uri, scope)
        if existing is not None:
            await self._remove_existing_chunks(existing.id)
            document = existing.model_copy(
                update={
                    "title": title,
                    "content_hash": content_hash,
                    "kind": kind,
                    "tags": list(tags or []),
                    "metadata": doc_metadata,
                    "updated_at": utcnow(),
                }
            )
        else:
            document = DocumentRecord(
                source_type="memory",
                source_uri=source_uri,
                title=title,
                content_hash=content_hash,
                scope=scope,
                kind=kind,
                tags=list(tags or []),
                metadata=doc_metadata,
            )
        await self._index_document(document, chunk_markdown(text), None)
        return document.id

    async def _remove_existing_chunks(self, document_id: str) -> None:
        embedding_ids = await self._metadata.delete_chunks_for_document(document_id)
        await self._vector.delete(embedding_ids)

    async def _index_document(
        self, document: DocumentRecord, drafts: list[ChunkDraft], path: str | None
    ) -> list[ChunkRecord]:
        await self._metadata.upsert_document(document)
        chunks = [
            ChunkRecord(
                document_id=document.id,
                ordinal=draft.ordinal,
                heading_path=draft.heading_path,
                text=draft.text,
                token_count=draft.token_count,
                scope=document.scope,
                kind=document.kind,
                repo=document.repo,
                path=path,
                language=document.language,
                tags=document.tags,
                source_type=document.source_type,
                metadata={
                    key: value for key, value in document.metadata.items() if key == "record_id"
                },
                embedding_id=new_id(),
            )
            for draft in drafts
        ]
        await self._metadata.insert_chunks(chunks)
        await self._keyword.index_chunks(chunks, document.title)
        if chunks:
            vectors = await self._embedder.embed_passages([c.text for c in chunks])
            points = [
                VectorPoint(
                    id=chunk.embedding_id or new_id(),
                    vector=vector,
                    payload=build_point_payload(
                        item_id=chunk.id,
                        item_kind="chunk",
                        document_id=document.id,
                        scope=chunk.scope,
                        kind=chunk.kind,
                        repo=chunk.repo,
                        path=chunk.path,
                        language=chunk.language,
                        tags=chunk.tags,
                        source_type=chunk.source_type,
                        created_at=chunk.created_at,
                        valid_from=chunk.valid_from,
                        valid_to=chunk.valid_to,
                    ),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            await self._vector.upsert(points)
        return chunks
