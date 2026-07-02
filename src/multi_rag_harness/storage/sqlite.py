"""SQLite storage adapter.

``SqliteStore`` implements both ``MetadataStore`` and ``KeywordIndex`` on one
database file so chunk rows and their FTS5 rows stay transactionally
consistent. List and dict fields are stored as JSON text; datetimes as
ISO-8601 UTC strings (lexicographically comparable).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from multi_rag_harness.storage.interfaces import (
    AliasRecord,
    ChunkRecord,
    ClaimRecord,
    DecisionRecord,
    DocumentRecord,
    ExtractionRunRecord,
    FailureRecord,
    ProvenanceRecord,
    ScoredId,
    SearchFilters,
    ToolRecord,
    TraceRecord,
    utcnow,
)

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_uri TEXT NOT NULL,
  title TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT 'default',
  kind TEXT NOT NULL DEFAULT 'doc',
  repo TEXT,
  language TEXT,
  tags TEXT NOT NULL DEFAULT '[]',
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (source_uri, scope)
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  heading_path TEXT NOT NULL DEFAULT '',
  text TEXT NOT NULL,
  token_count INTEGER NOT NULL,
  scope TEXT NOT NULL,
  kind TEXT NOT NULL,
  repo TEXT,
  path TEXT,
  language TEXT,
  tags TEXT NOT NULL DEFAULT '[]',
  source_type TEXT NOT NULL,
  valid_from TEXT,
  valid_to TEXT,
  metadata TEXT NOT NULL DEFAULT '{}',
  embedding_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kind_scope ON chunks(kind, scope);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED, text, heading_path, title
);

CREATE VIRTUAL TABLE IF NOT EXISTS graph_items_fts USING fts5(
  item_id UNINDEXED, item_type UNINDEXED, text
);

CREATE TABLE IF NOT EXISTS provenance (
  id TEXT PRIMARY KEY,
  item_type TEXT NOT NULL,
  item_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  chunk_id TEXT,
  evidence_text TEXT,
  char_start INTEGER,
  char_end INTEGER,
  extraction_run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_provenance_item ON provenance(item_id);
CREATE INDEX IF NOT EXISTS idx_provenance_chunk ON provenance(chunk_id);

CREATE TABLE IF NOT EXISTS extraction_runs (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  codex_thread_id TEXT,
  prompt_version TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  started_at TEXT,
  completed_at TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_extraction_status ON extraction_runs(status);

CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY,
  task TEXT NOT NULL,
  outcome TEXT NOT NULL,
  tools_used TEXT NOT NULL DEFAULT '[]',
  commands TEXT NOT NULL DEFAULT '[]',
  files_read TEXT NOT NULL DEFAULT '[]',
  files_changed TEXT NOT NULL DEFAULT '[]',
  errors TEXT NOT NULL DEFAULT '[]',
  tests TEXT NOT NULL DEFAULT '[]',
  final_response TEXT,
  human_feedback TEXT,
  linked_decisions TEXT NOT NULL DEFAULT '[]',
  linked_entities TEXT NOT NULL DEFAULT '[]',
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  status TEXT NOT NULL,
  context TEXT,
  decision TEXT NOT NULL,
  rationale TEXT,
  alternatives TEXT NOT NULL DEFAULT '[]',
  consequences TEXT,
  source_links TEXT NOT NULL DEFAULT '[]',
  related_entities TEXT NOT NULL DEFAULT '[]',
  supersedes TEXT,
  superseded_by TEXT,
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS failures (
  id TEXT PRIMARY KEY,
  error_text TEXT NOT NULL,
  error_category TEXT,
  command TEXT,
  environment TEXT,
  suspected_cause TEXT,
  confirmed_cause TEXT,
  fix_applied TEXT,
  verification TEXT,
  related_traces TEXT NOT NULL DEFAULT '[]',
  related_code_paths TEXT NOT NULL DEFAULT '[]',
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_entries (
  id TEXT PRIMARY KEY,
  server TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  input_schema TEXT NOT NULL DEFAULT '{}',
  output_shape TEXT,
  approval_policy TEXT,
  rate_limits TEXT,
  examples TEXT NOT NULL DEFAULT '[]',
  known_failure_modes TEXT NOT NULL DEFAULT '[]',
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (server, name)
);

CREATE TABLE IF NOT EXISTS claims (
  id TEXT PRIMARY KEY,
  subject_entity_id TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object_text TEXT NOT NULL,
  object_entity_id TEXT,
  modality TEXT NOT NULL,
  confidence REAL NOT NULL,
  valid_from TEXT,
  valid_to TEXT,
  metadata TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_claims_object ON claims(object_entity_id);

CREATE TABLE IF NOT EXISTS entity_aliases (
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  alias TEXT NOT NULL,
  normalized_alias TEXT NOT NULL,
  source_id TEXT,
  confidence REAL NOT NULL DEFAULT 1.0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_alias_norm ON entity_aliases(normalized_alias, entity_id);
CREATE INDEX IF NOT EXISTS idx_alias_lookup ON entity_aliases(normalized_alias);
"""


def build_fts_match(query: str) -> str:
    """Build a safe FTS5 MATCH expression from raw query text.

    Every whitespace token is double-quoted (neutralizing FTS5 operators such
    as AND/OR/NOT/NEAR/*/^/-/:) and tokens are OR-joined for recall; ranking
    layers downstream restore precision.
    """
    tokens = query.split()
    if not tokens:
        raise ValueError("query must not be empty")
    quoted = ['"' + token.replace('"', '""') + '"' for token in tokens]
    return " OR ".join(quoted)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _parse_dt_required(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _dump_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _dump_dict(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _placeholders(n: int) -> str:
    return ", ".join("?" * n)


def _filters_to_sql(filters: SearchFilters | None, *, alias: str = "c") -> tuple[str, list[Any]]:
    """Translate ``SearchFilters`` into an SQL fragment (ANDed) and params."""
    if filters is None:
        filters = SearchFilters()
    clauses: list[str] = []
    params: list[Any] = []
    if filters.scopes:
        clauses.append(f"{alias}.scope IN ({_placeholders(len(filters.scopes))})")
        params.extend(filters.scopes)
    if filters.kinds:
        clauses.append(f"{alias}.kind IN ({_placeholders(len(filters.kinds))})")
        params.extend(filters.kinds)
    if filters.repo is not None:
        clauses.append(f"{alias}.repo = ?")
        params.append(filters.repo)
    if filters.path_prefix is not None:
        clauses.append(f"{alias}.path LIKE ? ESCAPE '\\'")
        escaped = filters.path_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.append(escaped + "%")
    if filters.language is not None:
        clauses.append(f"{alias}.language = ?")
        params.append(filters.language)
    if filters.source_type is not None:
        clauses.append(f"{alias}.source_type = ?")
        params.append(filters.source_type)
    if filters.created_after is not None:
        clauses.append(f"{alias}.created_at >= ?")
        params.append(filters.created_after.isoformat())
    if filters.created_before is not None:
        clauses.append(f"{alias}.created_at <= ?")
        params.append(filters.created_before.isoformat())
    if filters.tags:
        clauses.append(
            f"EXISTS (SELECT 1 FROM json_each({alias}.tags) "
            f"WHERE json_each.value IN ({_placeholders(len(filters.tags))}))"
        )
        params.extend(filters.tags)
    if not filters.include_expired:
        valid_at = (filters.valid_at or utcnow()).isoformat()
        clauses.append(f"({alias}.valid_to IS NULL OR {alias}.valid_to >= ?)")
        params.append(valid_at)
        clauses.append(f"({alias}.valid_from IS NULL OR {alias}.valid_from <= ?)")
        params.append(valid_at)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


class SqliteStore:
    """MetadataStore + KeywordIndex over a single SQLite database."""

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteStore is not initialized; call initialize() first")
        return self._conn

    async def initialize(self) -> None:
        if self._conn is not None:
            return
        conn = await aiosqlite.connect(self._path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.executescript(_DDL)
        await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await conn.commit()
        self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # -- documents ---------------------------------------------------------

    async def upsert_document(self, doc: DocumentRecord) -> None:
        async with self._write_lock:
            await self.conn.execute(
                """
                INSERT INTO documents (id, source_type, source_uri, title, content_hash,
                    scope, kind, repo, language, tags, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_type=excluded.source_type, source_uri=excluded.source_uri,
                    title=excluded.title, content_hash=excluded.content_hash,
                    scope=excluded.scope, kind=excluded.kind, repo=excluded.repo,
                    language=excluded.language, tags=excluded.tags,
                    metadata=excluded.metadata, updated_at=excluded.updated_at
                """,
                (
                    doc.id,
                    doc.source_type,
                    doc.source_uri,
                    doc.title,
                    doc.content_hash,
                    doc.scope,
                    doc.kind,
                    doc.repo,
                    doc.language,
                    _dump_list(doc.tags),
                    _dump_dict(doc.metadata),
                    _iso(doc.created_at),
                    _iso(doc.updated_at),
                ),
            )
            await self.conn.commit()

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        cursor = await self.conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = await cursor.fetchone()
        return self._document_from_row(row) if row else None

    async def find_document_by_uri(self, source_uri: str, scope: str) -> DocumentRecord | None:
        cursor = await self.conn.execute(
            "SELECT * FROM documents WHERE source_uri = ? AND scope = ?", (source_uri, scope)
        )
        row = await cursor.fetchone()
        return self._document_from_row(row) if row else None

    async def delete_document(self, document_id: str) -> None:
        async with self._write_lock:
            cursor = await self.conn.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            )
            chunk_ids = [row["id"] for row in await cursor.fetchall()]
            if chunk_ids:
                await self.conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({_placeholders(len(chunk_ids))})",
                    chunk_ids,
                )
            await self.conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            await self.conn.commit()

    @staticmethod
    def _document_from_row(row: aiosqlite.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            source_type=row["source_type"],
            source_uri=row["source_uri"],
            title=row["title"],
            content_hash=row["content_hash"],
            scope=row["scope"],
            kind=row["kind"],
            repo=row["repo"],
            language=row["language"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            created_at=_parse_dt_required(row["created_at"]),
            updated_at=_parse_dt_required(row["updated_at"]),
        )

    # -- chunks --------------------------------------------------------------

    async def insert_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        if not chunks:
            return
        async with self._write_lock:
            await self.conn.executemany(
                """
                INSERT INTO chunks (id, document_id, ordinal, heading_path, text, token_count,
                    scope, kind, repo, path, language, tags, source_type, valid_from, valid_to,
                    metadata, embedding_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.id,
                        c.document_id,
                        c.ordinal,
                        c.heading_path,
                        c.text,
                        c.token_count,
                        c.scope,
                        c.kind,
                        c.repo,
                        c.path,
                        c.language,
                        _dump_list(c.tags),
                        c.source_type,
                        _iso(c.valid_from),
                        _iso(c.valid_to),
                        _dump_dict(c.metadata),
                        c.embedding_id,
                        _iso(c.created_at),
                        _iso(c.updated_at),
                    )
                    for c in chunks
                ],
            )
            await self.conn.commit()

    async def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        cursor = await self.conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = await cursor.fetchone()
        return self._chunk_from_row(row) if row else None

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        cursor = await self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({_placeholders(len(chunk_ids))})",
            list(chunk_ids),
        )
        rows = await cursor.fetchall()
        by_id = {row["id"]: self._chunk_from_row(row) for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    async def get_chunk_window(self, chunk_id: str, around: int) -> list[ChunkRecord]:
        target = await self.get_chunk(chunk_id)
        if target is None:
            return []
        cursor = await self.conn.execute(
            """
            SELECT * FROM chunks
            WHERE document_id = ? AND ordinal BETWEEN ? AND ?
            ORDER BY ordinal
            """,
            (target.document_id, target.ordinal - around, target.ordinal + around),
        )
        return [self._chunk_from_row(row) for row in await cursor.fetchall()]

    async def get_chunks_for_document(
        self, document_id: str, limit: int | None = None
    ) -> list[ChunkRecord]:
        sql = "SELECT * FROM chunks WHERE document_id = ? ORDER BY ordinal"
        params: list[Any] = [document_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        cursor = await self.conn.execute(sql, params)
        return [self._chunk_from_row(row) for row in await cursor.fetchall()]

    async def delete_chunks_for_document(self, document_id: str) -> list[str]:
        async with self._write_lock:
            cursor = await self.conn.execute(
                "SELECT id, embedding_id FROM chunks WHERE document_id = ?", (document_id,)
            )
            rows = await cursor.fetchall()
            chunk_ids = [row["id"] for row in rows]
            embedding_ids = [row["embedding_id"] for row in rows if row["embedding_id"]]
            if chunk_ids:
                await self.conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({_placeholders(len(chunk_ids))})",
                    chunk_ids,
                )
                await self.conn.execute(
                    f"DELETE FROM chunks WHERE id IN ({_placeholders(len(chunk_ids))})",
                    chunk_ids,
                )
            await self.conn.commit()
        return embedding_ids

    async def expire_chunks_for_document(self, document_id: str, valid_to: datetime) -> None:
        async with self._write_lock:
            await self.conn.execute(
                "UPDATE chunks SET valid_to = ?, updated_at = ? WHERE document_id = ?",
                (_iso(valid_to), _iso(utcnow()), document_id),
            )
            await self.conn.commit()

    @staticmethod
    def _chunk_from_row(row: aiosqlite.Row) -> ChunkRecord:
        return ChunkRecord(
            id=row["id"],
            document_id=row["document_id"],
            ordinal=row["ordinal"],
            heading_path=row["heading_path"],
            text=row["text"],
            token_count=row["token_count"],
            scope=row["scope"],
            kind=row["kind"],
            repo=row["repo"],
            path=row["path"],
            language=row["language"],
            tags=json.loads(row["tags"]),
            source_type=row["source_type"],
            valid_from=_parse_dt(row["valid_from"]),
            valid_to=_parse_dt(row["valid_to"]),
            metadata=json.loads(row["metadata"]),
            embedding_id=row["embedding_id"],
            created_at=_parse_dt_required(row["created_at"]),
            updated_at=_parse_dt_required(row["updated_at"]),
        )

    # -- keyword index (FTS5) ------------------------------------------------

    async def index_chunks(self, chunks: Sequence[ChunkRecord], title: str) -> None:
        if not chunks:
            return
        async with self._write_lock:
            await self.conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, text, heading_path, title) VALUES (?, ?, ?, ?)",
                [(c.id, c.text, c.heading_path, title) for c in chunks],
            )
            await self.conn.commit()

    async def remove_chunks(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        async with self._write_lock:
            await self.conn.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({_placeholders(len(chunk_ids))})",
                list(chunk_ids),
            )
            await self.conn.commit()

    async def search_chunks(
        self, query: str, filters: SearchFilters | None, limit: int
    ) -> list[ScoredId]:
        match = build_fts_match(query)
        filter_sql, filter_params = _filters_to_sql(filters)
        cursor = await self.conn.execute(
            f"""
            SELECT f.chunk_id AS chunk_id, -bm25(chunks_fts) AS score
            FROM chunks_fts f JOIN chunks c ON c.id = f.chunk_id
            WHERE chunks_fts MATCH ?{filter_sql}
            ORDER BY score DESC
            LIMIT ?
            """,
            [match, *filter_params, limit],
        )
        rows = await cursor.fetchall()
        return [ScoredId(id=row["chunk_id"], score=row["score"]) for row in rows]

    async def index_graph_item(self, item_id: str, item_type: str, text: str) -> None:
        async with self._write_lock:
            await self.conn.execute("DELETE FROM graph_items_fts WHERE item_id = ?", (item_id,))
            await self.conn.execute(
                "INSERT INTO graph_items_fts (item_id, item_type, text) VALUES (?, ?, ?)",
                (item_id, item_type, text),
            )
            await self.conn.commit()

    async def remove_graph_item(self, item_id: str) -> None:
        async with self._write_lock:
            await self.conn.execute("DELETE FROM graph_items_fts WHERE item_id = ?", (item_id,))
            await self.conn.commit()

    async def search_graph_items(self, query: str, item_type: str, limit: int) -> list[ScoredId]:
        match = build_fts_match(query)
        cursor = await self.conn.execute(
            """
            SELECT item_id, -bm25(graph_items_fts) AS score
            FROM graph_items_fts
            WHERE graph_items_fts MATCH ? AND item_type = ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (match, item_type, limit),
        )
        rows = await cursor.fetchall()
        return [ScoredId(id=row["item_id"], score=row["score"]) for row in rows]

    # -- provenance and extraction runs ---------------------------------------

    async def insert_provenance(self, records: Sequence[ProvenanceRecord]) -> None:
        if not records:
            return
        async with self._write_lock:
            await self.conn.executemany(
                """
                INSERT INTO provenance (id, item_type, item_id, source_id, chunk_id,
                    evidence_text, char_start, char_end, extraction_run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p.id,
                        p.item_type,
                        p.item_id,
                        p.source_id,
                        p.chunk_id,
                        p.evidence_text,
                        p.char_start,
                        p.char_end,
                        p.extraction_run_id,
                    )
                    for p in records
                ],
            )
            await self.conn.commit()

    async def get_provenance_for_item(self, item_id: str) -> list[ProvenanceRecord]:
        cursor = await self.conn.execute("SELECT * FROM provenance WHERE item_id = ?", (item_id,))
        return [self._provenance_from_row(row) for row in await cursor.fetchall()]

    async def get_entity_ids_for_chunks(self, chunk_ids: Sequence[str]) -> list[str]:
        if not chunk_ids:
            return []
        cursor = await self.conn.execute(
            f"""
            SELECT DISTINCT item_id FROM provenance
            WHERE item_type = 'entity' AND chunk_id IN ({_placeholders(len(chunk_ids))})
            """,
            list(chunk_ids),
        )
        return [row["item_id"] for row in await cursor.fetchall()]

    async def create_extraction_runs(self, runs: Sequence[ExtractionRunRecord]) -> None:
        if not runs:
            return
        async with self._write_lock:
            await self.conn.executemany(
                """
                INSERT INTO extraction_runs (id, source_id, chunk_id, codex_thread_id,
                    prompt_version, status, started_at, completed_at, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.id,
                        r.source_id,
                        r.chunk_id,
                        r.codex_thread_id,
                        r.prompt_version,
                        r.status,
                        _iso(r.started_at),
                        _iso(r.completed_at),
                        r.error,
                    )
                    for r in runs
                ],
            )
            await self.conn.commit()

    async def get_extraction_run(self, run_id: str) -> ExtractionRunRecord | None:
        cursor = await self.conn.execute("SELECT * FROM extraction_runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
        return self._extraction_run_from_row(row) if row else None

    async def claim_pending_extraction_runs(self, limit: int) -> list[ExtractionRunRecord]:
        async with self._write_lock:
            cursor = await self.conn.execute(
                "SELECT * FROM extraction_runs WHERE status = 'pending' ORDER BY rowid LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            if not rows:
                return []
            run_ids = [row["id"] for row in rows]
            started = utcnow()
            await self.conn.execute(
                f"""
                UPDATE extraction_runs SET status = 'running', started_at = ?
                WHERE id IN ({_placeholders(len(run_ids))})
                """,
                [_iso(started), *run_ids],
            )
            await self.conn.commit()
        runs = [self._extraction_run_from_row(row) for row in rows]
        for run in runs:
            run.status = "running"
            run.started_at = started
        return runs

    async def update_extraction_run(
        self,
        run_id: str,
        *,
        status: str,
        codex_thread_id: str | None = None,
        error: str | None = None,
    ) -> None:
        completed_at = utcnow() if status in ("completed", "failed") else None
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE extraction_runs
                SET status = ?,
                    codex_thread_id = COALESCE(?, codex_thread_id),
                    error = ?,
                    completed_at = COALESCE(?, completed_at)
                WHERE id = ?
                """,
                (status, codex_thread_id, error, _iso(completed_at), run_id),
            )
            await self.conn.commit()

    @staticmethod
    def _provenance_from_row(row: aiosqlite.Row) -> ProvenanceRecord:
        return ProvenanceRecord(
            id=row["id"],
            item_type=row["item_type"],
            item_id=row["item_id"],
            source_id=row["source_id"],
            chunk_id=row["chunk_id"],
            evidence_text=row["evidence_text"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            extraction_run_id=row["extraction_run_id"],
        )

    @staticmethod
    def _extraction_run_from_row(row: aiosqlite.Row) -> ExtractionRunRecord:
        return ExtractionRunRecord(
            id=row["id"],
            source_id=row["source_id"],
            chunk_id=row["chunk_id"],
            codex_thread_id=row["codex_thread_id"],
            prompt_version=row["prompt_version"],
            status=row["status"],
            started_at=_parse_dt(row["started_at"]),
            completed_at=_parse_dt(row["completed_at"]),
            error=row["error"],
        )

    # -- memory records --------------------------------------------------------

    async def insert_trace(self, record: TraceRecord) -> None:
        async with self._write_lock:
            await self.conn.execute(
                """
                INSERT INTO traces (id, task, outcome, tools_used, commands, files_read,
                    files_changed, errors, tests, final_response, human_feedback,
                    linked_decisions, linked_entities, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.task,
                    record.outcome,
                    _dump_list(record.tools_used),
                    _dump_list(record.commands),
                    _dump_list(record.files_read),
                    _dump_list(record.files_changed),
                    _dump_list(record.errors),
                    _dump_list(record.tests),
                    record.final_response,
                    record.human_feedback,
                    _dump_list(record.linked_decisions),
                    _dump_list(record.linked_entities),
                    _dump_dict(record.metadata),
                    _iso(record.created_at),
                ),
            )
            await self.conn.commit()

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        cursor = await self.conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return TraceRecord(
            id=row["id"],
            task=row["task"],
            outcome=row["outcome"],
            tools_used=json.loads(row["tools_used"]),
            commands=json.loads(row["commands"]),
            files_read=json.loads(row["files_read"]),
            files_changed=json.loads(row["files_changed"]),
            errors=json.loads(row["errors"]),
            tests=json.loads(row["tests"]),
            final_response=row["final_response"],
            human_feedback=row["human_feedback"],
            linked_decisions=json.loads(row["linked_decisions"]),
            linked_entities=json.loads(row["linked_entities"]),
            metadata=json.loads(row["metadata"]),
            created_at=_parse_dt_required(row["created_at"]),
        )

    async def insert_decision(self, record: DecisionRecord) -> None:
        async with self._write_lock:
            await self.conn.execute(
                """
                INSERT INTO decisions (id, title, status, context, decision, rationale,
                    alternatives, consequences, source_links, related_entities,
                    supersedes, superseded_by, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.title,
                    record.status,
                    record.context,
                    record.decision,
                    record.rationale,
                    _dump_list(record.alternatives),
                    record.consequences,
                    _dump_list(record.source_links),
                    _dump_list(record.related_entities),
                    record.supersedes,
                    record.superseded_by,
                    _dump_dict(record.metadata),
                    _iso(record.created_at),
                    _iso(record.updated_at),
                ),
            )
            await self.conn.commit()

    async def get_decision(self, decision_id: str) -> DecisionRecord | None:
        cursor = await self.conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return DecisionRecord(
            id=row["id"],
            title=row["title"],
            status=row["status"],
            context=row["context"],
            decision=row["decision"],
            rationale=row["rationale"],
            alternatives=json.loads(row["alternatives"]),
            consequences=row["consequences"],
            source_links=json.loads(row["source_links"]),
            related_entities=json.loads(row["related_entities"]),
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
            metadata=json.loads(row["metadata"]),
            created_at=_parse_dt_required(row["created_at"]),
            updated_at=_parse_dt_required(row["updated_at"]),
        )

    async def mark_decision_superseded(self, old_id: str, new_id: str) -> None:
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE decisions
                SET status = 'superseded', superseded_by = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_id, _iso(utcnow()), old_id),
            )
            await self.conn.commit()

    async def insert_failure(self, record: FailureRecord) -> None:
        async with self._write_lock:
            await self.conn.execute(
                """
                INSERT INTO failures (id, error_text, error_category, command, environment,
                    suspected_cause, confirmed_cause, fix_applied, verification,
                    related_traces, related_code_paths, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.error_text,
                    record.error_category,
                    record.command,
                    record.environment,
                    record.suspected_cause,
                    record.confirmed_cause,
                    record.fix_applied,
                    record.verification,
                    _dump_list(record.related_traces),
                    _dump_list(record.related_code_paths),
                    _dump_dict(record.metadata),
                    _iso(record.created_at),
                ),
            )
            await self.conn.commit()

    async def get_failure(self, failure_id: str) -> FailureRecord | None:
        cursor = await self.conn.execute("SELECT * FROM failures WHERE id = ?", (failure_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return FailureRecord(
            id=row["id"],
            error_text=row["error_text"],
            error_category=row["error_category"],
            command=row["command"],
            environment=row["environment"],
            suspected_cause=row["suspected_cause"],
            confirmed_cause=row["confirmed_cause"],
            fix_applied=row["fix_applied"],
            verification=row["verification"],
            related_traces=json.loads(row["related_traces"]),
            related_code_paths=json.loads(row["related_code_paths"]),
            metadata=json.loads(row["metadata"]),
            created_at=_parse_dt_required(row["created_at"]),
        )

    async def upsert_tool_record(self, record: ToolRecord) -> ToolRecord:
        async with self._write_lock:
            cursor = await self.conn.execute(
                "SELECT id, created_at FROM tool_entries WHERE server = ? AND name = ?",
                (record.server, record.name),
            )
            existing = await cursor.fetchone()
            stored = record
            if existing is not None:
                stored = record.model_copy(
                    update={
                        "id": existing["id"],
                        "created_at": _parse_dt(existing["created_at"]),
                        "updated_at": utcnow(),
                    }
                )
                await self.conn.execute(
                    """
                    UPDATE tool_entries
                    SET description = ?, input_schema = ?, output_shape = ?,
                        approval_policy = ?, rate_limits = ?, examples = ?,
                        known_failure_modes = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        stored.description,
                        _dump_dict(stored.input_schema),
                        stored.output_shape,
                        stored.approval_policy,
                        stored.rate_limits,
                        _dump_list(stored.examples),
                        _dump_list(stored.known_failure_modes),
                        _dump_dict(stored.metadata),
                        _iso(stored.updated_at),
                        stored.id,
                    ),
                )
            else:
                await self.conn.execute(
                    """
                    INSERT INTO tool_entries (id, server, name, description, input_schema,
                        output_shape, approval_policy, rate_limits, examples,
                        known_failure_modes, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stored.id,
                        stored.server,
                        stored.name,
                        stored.description,
                        _dump_dict(stored.input_schema),
                        stored.output_shape,
                        stored.approval_policy,
                        stored.rate_limits,
                        _dump_list(stored.examples),
                        _dump_list(stored.known_failure_modes),
                        _dump_dict(stored.metadata),
                        _iso(stored.created_at),
                        _iso(stored.updated_at),
                    ),
                )
            await self.conn.commit()
        return stored

    async def get_tool_record(self, tool_id: str) -> ToolRecord | None:
        cursor = await self.conn.execute("SELECT * FROM tool_entries WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return ToolRecord(
            id=row["id"],
            server=row["server"],
            name=row["name"],
            description=row["description"],
            input_schema=json.loads(row["input_schema"]),
            output_shape=row["output_shape"],
            approval_policy=row["approval_policy"],
            rate_limits=row["rate_limits"],
            examples=json.loads(row["examples"]),
            known_failure_modes=json.loads(row["known_failure_modes"]),
            metadata=json.loads(row["metadata"]),
            created_at=_parse_dt_required(row["created_at"]),
            updated_at=_parse_dt_required(row["updated_at"]),
        )

    # -- graph relational data ---------------------------------------------------

    async def insert_claims(self, claims: Sequence[ClaimRecord]) -> None:
        if not claims:
            return
        async with self._write_lock:
            await self.conn.executemany(
                """
                INSERT INTO claims (id, subject_entity_id, predicate, object_text,
                    object_entity_id, modality, confidence, valid_from, valid_to,
                    metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.id,
                        c.subject_entity_id,
                        c.predicate,
                        c.object_text,
                        c.object_entity_id,
                        c.modality,
                        c.confidence,
                        _iso(c.valid_from),
                        _iso(c.valid_to),
                        _dump_dict(c.metadata),
                        _iso(c.created_at),
                    )
                    for c in claims
                ],
            )
            await self.conn.commit()

    async def get_claim(self, claim_id: str) -> ClaimRecord | None:
        results = await self.get_claims([claim_id])
        return results[0] if results else None

    async def get_claims(self, claim_ids: Sequence[str]) -> list[ClaimRecord]:
        if not claim_ids:
            return []
        cursor = await self.conn.execute(
            f"SELECT * FROM claims WHERE id IN ({_placeholders(len(claim_ids))})",
            list(claim_ids),
        )
        rows = await cursor.fetchall()
        by_id = {row["id"]: self._claim_from_row(row) for row in rows}
        return [by_id[cid] for cid in claim_ids if cid in by_id]

    async def get_claims_for_entity(
        self, entity_id: str, valid_at: datetime | None = None, limit: int = 20
    ) -> list[ClaimRecord]:
        at = (valid_at or utcnow()).isoformat()
        cursor = await self.conn.execute(
            """
            SELECT * FROM claims
            WHERE (subject_entity_id = ? OR object_entity_id = ?)
              AND (valid_to IS NULL OR valid_to >= ?)
              AND (valid_from IS NULL OR valid_from <= ?)
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            (entity_id, entity_id, at, at, limit),
        )
        return [self._claim_from_row(row) for row in await cursor.fetchall()]

    @staticmethod
    def _claim_from_row(row: aiosqlite.Row) -> ClaimRecord:
        return ClaimRecord(
            id=row["id"],
            subject_entity_id=row["subject_entity_id"],
            predicate=row["predicate"],
            object_text=row["object_text"],
            object_entity_id=row["object_entity_id"],
            modality=row["modality"],
            confidence=row["confidence"],
            valid_from=_parse_dt(row["valid_from"]),
            valid_to=_parse_dt(row["valid_to"]),
            metadata=json.loads(row["metadata"]),
            created_at=_parse_dt_required(row["created_at"]),
        )

    async def insert_aliases(self, aliases: Sequence[AliasRecord]) -> None:
        if not aliases:
            return
        async with self._write_lock:
            await self.conn.executemany(
                """
                INSERT INTO entity_aliases (id, entity_id, alias, normalized_alias,
                    source_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_alias, entity_id) DO NOTHING
                """,
                [
                    (a.id, a.entity_id, a.alias, a.normalized_alias, a.source_id, a.confidence)
                    for a in aliases
                ],
            )
            await self.conn.commit()

    async def get_aliases_for_entity(self, entity_id: str) -> list[AliasRecord]:
        cursor = await self.conn.execute(
            "SELECT * FROM entity_aliases WHERE entity_id = ?", (entity_id,)
        )
        return [
            AliasRecord(
                id=row["id"],
                entity_id=row["entity_id"],
                alias=row["alias"],
                normalized_alias=row["normalized_alias"],
                source_id=row["source_id"],
                confidence=row["confidence"],
            )
            for row in await cursor.fetchall()
        ]

    async def find_entity_id_by_normalized_alias(self, normalized: str) -> str | None:
        cursor = await self.conn.execute(
            """
            SELECT entity_id FROM entity_aliases
            WHERE normalized_alias = ?
            ORDER BY confidence DESC
            LIMIT 1
            """,
            (normalized,),
        )
        row = await cursor.fetchone()
        return row["entity_id"] if row else None
