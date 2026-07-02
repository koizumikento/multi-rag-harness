"""Codex SDK graph extraction workflows.

The orchestrator drains pending extraction runs: prompt Codex per chunk,
validate the structured output, canonicalize entities, and persist graph
items with provenance. It is invoked explicitly (CLI ``extract`` or
orchestrator call), never implicitly during an MCP tool call.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from pydantic import BaseModel

from multi_rag_harness.codex.client import CodexClient, CodexRunError
from multi_rag_harness.codex.prompts import EXTRACTION_SCHEMA, build_extraction_prompt
from multi_rag_harness.config import Settings
from multi_rag_harness.graph.canonicalize import EntityCanonicalizer, normalize_name
from multi_rag_harness.graph.extraction import ExtractionValidationError, validate_extraction
from multi_rag_harness.graph.models import ValidatedExtraction
from multi_rag_harness.graph.traversal import GraphIndexer
from multi_rag_harness.storage.interfaces import (
    ChunkRecord,
    ClaimRecord,
    ExtractionRunRecord,
    GraphStore,
    MetadataStore,
    ProvenanceRecord,
    RelationEdge,
)

logger = logging.getLogger(__name__)


class ExtractionSummary(BaseModel):
    runs_attempted: int = 0
    runs_completed: int = 0
    runs_failed: int = 0
    entities_created: int = 0
    entities_merged: int = 0
    relations_created: int = 0
    claims_created: int = 0


def _parse_temporal(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _char_span(chunk_text: str, evidence: str) -> tuple[int | None, int | None]:
    start = chunk_text.find(evidence)
    if start < 0:
        return None, None
    return start, start + len(evidence)


class ExtractionOrchestrator:
    def __init__(
        self,
        codex: CodexClient,
        metadata: MetadataStore,
        graph: GraphStore,
        canonicalizer: EntityCanonicalizer,
        indexer: GraphIndexer,
        settings: Settings,
    ) -> None:
        self._codex = codex
        self._metadata = metadata
        self._graph = graph
        self._canonicalizer = canonicalizer
        self._indexer = indexer
        self._settings = settings

    async def run_pending(self, limit: int | None = None) -> ExtractionSummary:
        batch_limit = limit or self._settings.codex.max_runs_per_batch
        runs = await self._metadata.claim_pending_extraction_runs(batch_limit)
        summary = ExtractionSummary(runs_attempted=len(runs))
        for run in runs:
            try:
                await self._run_one(run, summary)
                summary.runs_completed += 1
            except (CodexRunError, ExtractionValidationError) as exc:
                logger.warning("extraction run %s failed: %s", run.id, exc)
                await self._metadata.update_extraction_run(run.id, status="failed", error=str(exc))
                summary.runs_failed += 1
        return summary

    async def _run_one(self, run: ExtractionRunRecord, summary: ExtractionSummary) -> None:
        chunk = await self._metadata.get_chunk(run.chunk_id)
        if chunk is None:
            raise ExtractionValidationError(f"chunk not found: {run.chunk_id}")
        document = await self._metadata.get_document(run.source_id)
        prompt = build_extraction_prompt(
            chunk.text,
            heading_path=chunk.heading_path,
            document_title=document.title if document else "",
        )
        raw, thread_id = await self._codex.run_structured(prompt, EXTRACTION_SCHEMA)
        validated = validate_extraction(raw, chunk.text)
        warnings = list(validated.warnings)

        name_to_id = await self._persist_entities(run, chunk, validated, summary)
        warnings.extend(await self._persist_relations(run, chunk, validated, name_to_id, summary))
        warnings.extend(await self._persist_claims(run, chunk, validated, name_to_id, summary))

        error = json.dumps({"warnings": warnings}, ensure_ascii=False) if warnings else None
        await self._metadata.update_extraction_run(
            run.id, status="completed", codex_thread_id=thread_id, error=error
        )

    async def _persist_entities(
        self,
        run: ExtractionRunRecord,
        chunk: ChunkRecord,
        validated: ValidatedExtraction,
        summary: ExtractionSummary,
    ) -> dict[str, str]:
        name_to_id: dict[str, str] = {}
        for extracted in validated.result.entities:
            entity_id, created = await self._canonicalizer.resolve(
                extracted, source_id=run.source_id
            )
            if created:
                summary.entities_created += 1
            else:
                summary.entities_merged += 1
            for name in (extracted.name, *extracted.aliases):
                normalized = normalize_name(name)
                if normalized:
                    name_to_id.setdefault(normalized, entity_id)
            char_start, char_end = _char_span(chunk.text, extracted.evidence)
            await self._metadata.insert_provenance(
                [
                    ProvenanceRecord(
                        item_type="entity",
                        item_id=entity_id,
                        source_id=run.source_id,
                        chunk_id=run.chunk_id,
                        evidence_text=extracted.evidence,
                        char_start=char_start,
                        char_end=char_end,
                        extraction_run_id=run.id,
                    )
                ]
            )
            entity = await self._graph.get_entity(entity_id)
            if entity is not None:
                aliases = await self._metadata.get_aliases_for_entity(entity_id)
                await self._indexer.index_entity(entity, [a.alias for a in aliases])
        return name_to_id

    async def _resolve_name(self, name: str, name_to_id: dict[str, str]) -> str | None:
        normalized = normalize_name(name)
        if not normalized:
            return None
        if normalized in name_to_id:
            return name_to_id[normalized]
        return await self._metadata.find_entity_id_by_normalized_alias(normalized)

    async def _persist_relations(
        self,
        run: ExtractionRunRecord,
        chunk: ChunkRecord,
        validated: ValidatedExtraction,
        name_to_id: dict[str, str],
        summary: ExtractionSummary,
    ) -> list[str]:
        warnings: list[str] = []
        for extracted in validated.result.relations:
            source_id = await self._resolve_name(extracted.source, name_to_id)
            target_id = await self._resolve_name(extracted.target, name_to_id)
            if source_id is None or target_id is None:
                warnings.append(
                    f"relation skipped, unresolved endpoint: "
                    f"{extracted.source} -[{extracted.type}]-> {extracted.target}"
                )
                continue
            relation = RelationEdge(
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=extracted.type,
                description=extracted.description,
                confidence=extracted.confidence,
                valid_from=_parse_temporal(extracted.valid_from),
                valid_to=_parse_temporal(extracted.valid_to),
            )
            await self._graph.upsert_relation(relation)
            char_start, char_end = _char_span(chunk.text, extracted.evidence)
            await self._metadata.insert_provenance(
                [
                    ProvenanceRecord(
                        item_type="relation",
                        item_id=relation.id,
                        source_id=run.source_id,
                        chunk_id=run.chunk_id,
                        evidence_text=extracted.evidence,
                        char_start=char_start,
                        char_end=char_end,
                        extraction_run_id=run.id,
                    )
                ]
            )
            summary.relations_created += 1
        return warnings

    async def _persist_claims(
        self,
        run: ExtractionRunRecord,
        chunk: ChunkRecord,
        validated: ValidatedExtraction,
        name_to_id: dict[str, str],
        summary: ExtractionSummary,
    ) -> list[str]:
        warnings: list[str] = []
        for extracted in validated.result.claims:
            subject_id = await self._resolve_name(extracted.subject, name_to_id)
            if subject_id is None:
                warnings.append(f"claim skipped, unresolved subject: {extracted.subject}")
                continue
            object_id = await self._resolve_name(extracted.object, name_to_id)
            claim = ClaimRecord(
                subject_entity_id=subject_id,
                predicate=extracted.predicate,
                object_text=extracted.object,
                object_entity_id=object_id,
                modality=extracted.modality,
                confidence=extracted.confidence,
                valid_from=_parse_temporal(extracted.valid_from),
                valid_to=_parse_temporal(extracted.valid_to),
            )
            await self._metadata.insert_claims([claim])
            char_start, char_end = _char_span(chunk.text, extracted.evidence)
            await self._metadata.insert_provenance(
                [
                    ProvenanceRecord(
                        item_type="claim",
                        item_id=claim.id,
                        source_id=run.source_id,
                        chunk_id=run.chunk_id,
                        evidence_text=extracted.evidence,
                        char_start=char_start,
                        char_end=char_end,
                        extraction_run_id=run.id,
                    )
                ]
            )
            subject = await self._graph.get_entity(subject_id)
            await self._indexer.index_claim(
                claim, subject.canonical_name if subject else extracted.subject
            )
            summary.claims_created += 1
        return warnings
