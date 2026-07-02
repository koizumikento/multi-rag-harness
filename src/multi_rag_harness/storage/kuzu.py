"""Kuzu graph storage adapter.

Stores entities, relations, and community membership for traversal. Claims,
aliases, and provenance live in the metadata store. The synchronous kuzu API
is wrapped in ``anyio.to_thread.run_sync``.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import anyio.to_thread
import kuzu

from multi_rag_harness.storage.interfaces import (
    CommunityNode,
    EntityNode,
    RelationEdge,
    utcnow,
)

T = TypeVar("T")

_DDL_STATEMENTS = [
    """
    CREATE NODE TABLE IF NOT EXISTS Entity(
        id STRING PRIMARY KEY,
        canonical_name STRING,
        entity_type STRING,
        description STRING,
        metadata STRING,
        created_at STRING,
        updated_at STRING
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS Community(
        id STRING PRIMARY KEY,
        title STRING,
        summary STRING,
        level INT64,
        created_at STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS RELATED_TO(
        FROM Entity TO Entity,
        id STRING,
        relation_type STRING,
        description STRING,
        confidence DOUBLE,
        valid_from STRING,
        valid_to STRING,
        metadata STRING,
        created_at STRING
    )
    """,
    "CREATE REL TABLE IF NOT EXISTS MEMBER_OF(FROM Entity TO Community)",
]

_ENTITY_COLUMNS = "id, canonical_name, entity_type, description, metadata, created_at, updated_at"
_RELATION_COLUMNS = (
    "r.id, r.relation_type, r.description, r.confidence, r.valid_from, r.valid_to, "
    "r.metadata, r.created_at"
)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _rows(result: Any) -> list[list[Any]]:
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


class KuzuGraphStore:
    """GraphStore over an embedded Kuzu database."""

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    async def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

    @property
    def conn(self) -> kuzu.Connection:
        if self._conn is None:
            raise RuntimeError("KuzuGraphStore is not initialized; call initialize() first")
        return self._conn

    async def initialize(self) -> None:
        if self._conn is not None:
            return

        def _init() -> tuple[kuzu.Database, kuzu.Connection]:
            db = kuzu.Database(self._path)
            conn = kuzu.Connection(db)
            for statement in _DDL_STATEMENTS:
                conn.execute(statement)
            return db, conn

        self._db, self._conn = await anyio.to_thread.run_sync(_init)

    async def close(self) -> None:
        def _close() -> None:
            if self._conn is not None:
                self._conn.close()
            if self._db is not None:
                self._db.close()

        await anyio.to_thread.run_sync(_close)
        self._conn = None
        self._db = None

    # -- entities -----------------------------------------------------------

    async def upsert_entity(self, entity: EntityNode) -> None:
        await self._run(
            self.conn.execute,
            """
            MERGE (e:Entity {id: $id})
            ON CREATE SET e.canonical_name = $canonical_name, e.entity_type = $entity_type,
                e.description = $description, e.metadata = $metadata,
                e.created_at = $created_at, e.updated_at = $updated_at
            ON MATCH SET e.canonical_name = $canonical_name, e.entity_type = $entity_type,
                e.description = $description, e.metadata = $metadata,
                e.updated_at = $updated_at
            """,
            {
                "id": entity.id,
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "metadata": json.dumps(entity.metadata, ensure_ascii=False),
                "created_at": _iso(entity.created_at),
                "updated_at": _iso(utcnow()),
            },
        )

    async def get_entity(self, entity_id: str) -> EntityNode | None:
        entities = await self.get_entities([entity_id])
        return entities[0] if entities else None

    async def get_entities(self, entity_ids: Sequence[str]) -> list[EntityNode]:
        if not entity_ids:
            return []
        result = await self._run(
            self.conn.execute,
            f"MATCH (e:Entity) WHERE e.id IN $ids "
            f"RETURN {', '.join('e.' + c.strip() for c in _ENTITY_COLUMNS.split(','))}",
            {"ids": list(entity_ids)},
        )
        by_id = {}
        for row in _rows(result):
            entity = self._entity_from_row(row)
            by_id[entity.id] = entity
        return [by_id[eid] for eid in entity_ids if eid in by_id]

    @staticmethod
    def _entity_from_row(row: list[Any]) -> EntityNode:
        return EntityNode(
            id=row[0],
            canonical_name=row[1],
            entity_type=row[2] or "",
            description=row[3] or "",
            metadata=json.loads(row[4]) if row[4] else {},
            created_at=_parse_dt(row[5]) or utcnow(),
            updated_at=_parse_dt(row[6]) or utcnow(),
        )

    # -- relations ----------------------------------------------------------

    async def upsert_relation(self, relation: RelationEdge) -> None:
        await self._run(
            self.conn.execute,
            """
            MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
            MERGE (a)-[r:RELATED_TO {id: $id}]->(b)
            ON CREATE SET r.relation_type = $relation_type, r.description = $description,
                r.confidence = $confidence, r.valid_from = $valid_from,
                r.valid_to = $valid_to, r.metadata = $metadata, r.created_at = $created_at
            ON MATCH SET r.relation_type = $relation_type, r.description = $description,
                r.confidence = $confidence, r.valid_from = $valid_from,
                r.valid_to = $valid_to, r.metadata = $metadata
            """,
            {
                "source_id": relation.source_entity_id,
                "target_id": relation.target_entity_id,
                "id": relation.id,
                "relation_type": relation.relation_type,
                "description": relation.description,
                "confidence": relation.confidence,
                "valid_from": _iso(relation.valid_from),
                "valid_to": _iso(relation.valid_to),
                "metadata": json.dumps(relation.metadata, ensure_ascii=False),
                "created_at": _iso(relation.created_at),
            },
        )

    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: Sequence[str] | None = None,
        valid_at: datetime | None = None,
    ) -> list[tuple[RelationEdge, EntityNode]]:
        entity_cols = ", ".join("n." + c.strip() for c in _ENTITY_COLUMNS.split(","))
        outgoing = await self._run(
            self.conn.execute,
            f"""
            MATCH (a:Entity {{id: $id}})-[r:RELATED_TO]->(n:Entity)
            RETURN {_RELATION_COLUMNS}, {entity_cols}
            """,
            {"id": entity_id},
        )
        incoming = await self._run(
            self.conn.execute,
            f"""
            MATCH (a:Entity {{id: $id}})<-[r:RELATED_TO]-(n:Entity)
            RETURN {_RELATION_COLUMNS}, {entity_cols}
            """,
            {"id": entity_id},
        )
        at = valid_at or utcnow()
        wanted = set(relation_types) if relation_types else None
        neighbors: list[tuple[RelationEdge, EntityNode]] = []
        seen_relation_ids: set[str] = set()
        tagged_rows = [(row, True) for row in _rows(outgoing)] + [
            (row, False) for row in _rows(incoming)
        ]
        for row, is_outgoing in tagged_rows:
            neighbor = self._entity_from_row(row[8:15])
            relation = RelationEdge(
                id=row[0],
                relation_type=row[1],
                description=row[2] or "",
                confidence=row[3],
                valid_from=_parse_dt(row[4]),
                valid_to=_parse_dt(row[5]),
                metadata=json.loads(row[6]) if row[6] else {},
                created_at=_parse_dt(row[7]) or utcnow(),
                source_entity_id=entity_id if is_outgoing else neighbor.id,
                target_entity_id=neighbor.id if is_outgoing else entity_id,
            )
            if relation.id in seen_relation_ids:
                continue
            seen_relation_ids.add(relation.id)
            if wanted is not None and relation.relation_type not in wanted:
                continue
            if relation.valid_from is not None and relation.valid_from > at:
                continue
            if relation.valid_to is not None and relation.valid_to < at:
                continue
            neighbors.append((relation, neighbor))
        return neighbors

    # -- communities ----------------------------------------------------------

    async def upsert_community(
        self, community: CommunityNode, member_entity_ids: Sequence[str]
    ) -> None:
        await self._run(
            self.conn.execute,
            """
            MERGE (c:Community {id: $id})
            ON CREATE SET c.title = $title, c.summary = $summary, c.level = $level,
                c.created_at = $created_at
            ON MATCH SET c.title = $title, c.summary = $summary, c.level = $level
            """,
            {
                "id": community.id,
                "title": community.title,
                "summary": community.summary,
                "level": community.level,
                "created_at": _iso(community.created_at),
            },
        )
        for entity_id in member_entity_ids:
            await self._run(
                self.conn.execute,
                """
                MATCH (e:Entity {id: $entity_id}), (c:Community {id: $community_id})
                MERGE (e)-[:MEMBER_OF]->(c)
                """,
                {"entity_id": entity_id, "community_id": community.id},
            )

    async def get_communities_for_entity(self, entity_id: str) -> list[CommunityNode]:
        result = await self._run(
            self.conn.execute,
            """
            MATCH (e:Entity {id: $id})-[:MEMBER_OF]->(c:Community)
            RETURN c.id, c.title, c.summary, c.level, c.created_at
            """,
            {"id": entity_id},
        )
        return [self._community_from_row(row) for row in _rows(result)]

    async def list_communities(self, limit: int = 50) -> list[CommunityNode]:
        result = await self._run(
            self.conn.execute,
            """
            MATCH (c:Community)
            RETURN c.id, c.title, c.summary, c.level, c.created_at
            ORDER BY c.level, c.title
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [self._community_from_row(row) for row in _rows(result)]

    @staticmethod
    def _community_from_row(row: list[Any]) -> CommunityNode:
        return CommunityNode(
            id=row[0],
            title=row[1],
            summary=row[2] or "",
            level=row[3],
            created_at=_parse_dt(row[4]) or utcnow(),
        )
