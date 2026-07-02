# Storage

This document describes the concrete storage backends, schemas, and adapter
boundaries. For product scope, see [specification.md](specification.md).

## Backend Selection

Backends sit behind four protocols in
`multi_rag_harness/storage/interfaces.py` and are selected via configuration
(`[storage]` in TOML or `MRH_STORAGE__*` env vars). `build_storage(settings)`
in `storage/__init__.py` returns a `StorageBundle(metadata, keyword, vector,
graph)`.

| Role | Protocol | Default backend | Alternative |
|------|----------|-----------------|-------------|
| Metadata + provenance + memory records | `MetadataStore` | SQLite (`SqliteStore`) | Postgres (stub, not implemented) |
| Keyword index | `KeywordIndex` | SQLite FTS5 (same `SqliteStore` instance) | Postgres FTS (stub) |
| Vector index | `VectorIndex` | Qdrant embedded local mode (`QdrantVectorIndex`) | pgvector (stub) |
| Graph store | `GraphStore` | Kuzu (`KuzuGraphStore`) | — |

The default configuration is fully local and server-free. All adapters are
async; the synchronous Qdrant/Kuzu/sentence-transformers calls are wrapped in
`anyio.to_thread.run_sync` inside their adapters.

Data lives under `data_dir` (default `./.multi-rag-harness`, override with
`MRH_DATA_DIR`): `metadata.db` (SQLite), `qdrant/` (embedded Qdrant), `kuzu/`
(Kuzu database). The directory is per-project memory and must be gitignored.

## SQLite Schema

`SqliteStore` implements both `MetadataStore` and `KeywordIndex` over one
database file so chunk rows and FTS rows stay consistent. Conventions:

- ids are uuid4 strings; timestamps are ISO-8601 UTC TEXT (lexicographically
  comparable); list/dict fields are JSON TEXT.
- WAL mode, foreign keys on. DDL is idempotent (`CREATE ... IF NOT EXISTS`)
  and stamps `PRAGMA user_version = 1`. Future schema migrations key off
  `user_version`.

Tables (see `storage/sqlite.py` for full DDL):

- `documents` — source_type (`file` | `memory` | `graph`), source_uri, title,
  content_hash, scope, kind, repo, language, tags, metadata, timestamps.
  Unique on `(source_uri, scope)`.
- `chunks` — document_id (FK, cascade), ordinal, heading_path, text,
  token_count, denormalized filter columns (scope, kind, repo, path, language,
  tags, source_type, valid_from, valid_to), embedding_id (Qdrant point id),
  metadata. Unique on `(document_id, ordinal)`.
- `chunks_fts` (FTS5) — chunk_id (unindexed), text, heading_path, title.
  Ranked with `-bm25(chunks_fts)`.
- `graph_items_fts` (FTS5) — item_id, item_type (`entity` | `claim`), text.
  Makes graph items keyword-searchable without turning them into documents.
- `provenance` — item_type/item_id → source_id (document), chunk_id,
  evidence_text, char offsets, extraction_run_id. Every graph item keeps at
  least one provenance row.
- `extraction_runs` — per-chunk Codex extraction jobs with status
  `pending → running → completed | failed`. Claiming is atomic under the
  store's write lock.
- `traces`, `decisions`, `failures`, `tool_entries` — typed memory records
  exactly as in the specification data model. `tool_entries` is unique on
  `(server, name)` and upserted. `decisions` carries
  `supersedes`/`superseded_by`.
- `claims`, `entity_aliases` — graph relational data. Aliases store
  `normalized_alias` (NFKC + casefold + whitespace collapse) with a unique
  `(normalized_alias, entity_id)` index for canonicalization lookups.

Metadata filters map to SQL via a shared builder: scope/kind `IN`, repo/
language/source_type equality, `path LIKE prefix%`, created_at range, tags via
`json_each` (OR semantics), and temporal validity
(`valid_to IS NULL OR valid_to >= valid_at`, same for `valid_from`) unless
`include_expired` is set.

FTS query safety: raw queries are tokenized on whitespace, each token is
double-quoted (neutralizing FTS5 operators), and tokens are OR-joined
(`build_fts_match`). Recall-oriented; RRF and reranking restore precision.

## Qdrant (Vector Index)

Embedded local mode by default (`QdrantClient(path=data_dir/qdrant)`);
setting `storage.qdrant_url` (env `MRH_STORAGE__QDRANT_URL`, key via
`MRH_STORAGE__QDRANT_API_KEY`) switches to a remote Qdrant server. One
collection (default `mrh_vectors`, cosine distance, size = embedding
dimension) holds chunks and graph items, separated by the `item_kind` payload
field (`chunk` | `entity` | `claim`).

On startup the adapter validates the configured `embedding.dimension` against
an existing collection and raises `VectorDimensionMismatchError` on conflict:
switching embedding models requires re-ingesting into a fresh data_dir or
collection — there is no vector migration.

Point payload:

```json
{"item_id": "...", "item_kind": "chunk", "document_id": "...",
 "scope": "...", "kind": "...", "repo": null, "path": null, "language": null,
 "tags": [], "source_type": "file",
 "created_at": 0.0, "valid_from": null, "valid_to": null}
```

Timestamps are epoch floats (Qdrant range filters). Point ids are uuid4
strings: `chunk.embedding_id` for chunks, the item's own id for graph items.
Search always returns `payload["item_id"]`, never the point id. `path_prefix`
and `confidence_min` are post-filtered in Python; everything else maps to
Qdrant `Filter` conditions. Temporal staleness in vector payloads (e.g. chunks
expired in place) is tolerated: the hybrid retriever re-checks validity after
hydrating chunks from SQLite.

## Kuzu (Graph Store)

Kuzu holds only what benefits from traversal; claims/aliases/provenance stay
in SQLite.

```cypher
CREATE NODE TABLE Entity(id STRING PRIMARY KEY, canonical_name STRING,
  entity_type STRING, description STRING, metadata STRING,
  created_at STRING, updated_at STRING);
CREATE NODE TABLE Community(id STRING PRIMARY KEY, title STRING,
  summary STRING, level INT64, created_at STRING);
CREATE REL TABLE RELATED_TO(FROM Entity TO Entity, id STRING,
  relation_type STRING, description STRING, confidence DOUBLE,
  valid_from STRING, valid_to STRING, metadata STRING, created_at STRING);
CREATE REL TABLE MEMBER_OF(FROM Entity TO Community);
```

Upserts use `MERGE ... ON CREATE SET ... ON MATCH SET`. Neighborhood expansion
runs two directed one-hop queries per node (outgoing/incoming) with Python-side
relation-type and temporal filtering; depth-N expansion is a Python BFS
(depth clamped 1–3, capped at 25 entities / 50 relations / 20 claims).

TODO: community detection (e.g. Leiden over the entity graph) is not
implemented. `CommunityService` only stores and retrieves externally provided
communities and summaries.

## Postgres / pgvector

`PostgresStore` and `PgvectorIndex` are selectable placeholders whose
operations raise `NotImplementedError`. The sqlalchemy/psycopg/pgvector
dependencies are retained for these future adapters.
