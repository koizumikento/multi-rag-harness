"""Qdrant vector storage adapter.

Uses the embedded local mode (``QdrantClient(path=...)``); no server needed.
The synchronous client is wrapped in ``anyio.to_thread.run_sync`` so the
adapter satisfies the async ``VectorIndex`` protocol.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import anyio.to_thread
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from multi_rag_harness.storage.interfaces import ScoredId, SearchFilters, VectorPoint, utcnow

T = TypeVar("T")

_PAYLOAD_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "item_kind": qm.PayloadSchemaType.KEYWORD,
    "scope": qm.PayloadSchemaType.KEYWORD,
    "kind": qm.PayloadSchemaType.KEYWORD,
    "repo": qm.PayloadSchemaType.KEYWORD,
    "tags": qm.PayloadSchemaType.KEYWORD,
    "document_id": qm.PayloadSchemaType.KEYWORD,
}


def _epoch(dt: datetime | None) -> float | None:
    return dt.timestamp() if dt is not None else None


def build_point_payload(
    *,
    item_id: str,
    item_kind: str,
    document_id: str | None = None,
    scope: str = "default",
    kind: str = "doc",
    repo: str | None = None,
    path: str | None = None,
    language: str | None = None,
    tags: Sequence[str] = (),
    source_type: str = "file",
    created_at: datetime | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> dict[str, Any]:
    """Canonical payload shape shared by chunk and graph-item points."""
    return {
        "item_id": item_id,
        "item_kind": item_kind,
        "document_id": document_id,
        "scope": scope,
        "kind": kind,
        "repo": repo,
        "path": path,
        "language": language,
        "tags": list(tags),
        "source_type": source_type,
        "created_at": _epoch(created_at or utcnow()),
        "valid_from": _epoch(valid_from),
        "valid_to": _epoch(valid_to),
    }


def _filters_to_qdrant(
    filters: SearchFilters | None, item_kinds: Sequence[str] | None
) -> qm.Filter | None:
    must: list[qm.Condition] = []
    should: list[qm.Condition] = []
    kinds = list(item_kinds) if item_kinds else ["chunk"]
    must.append(qm.FieldCondition(key="item_kind", match=qm.MatchAny(any=kinds)))
    if filters is not None:
        if filters.scopes:
            must.append(qm.FieldCondition(key="scope", match=qm.MatchAny(any=filters.scopes)))
        if filters.kinds:
            must.append(qm.FieldCondition(key="kind", match=qm.MatchAny(any=filters.kinds)))
        if filters.repo is not None:
            must.append(qm.FieldCondition(key="repo", match=qm.MatchValue(value=filters.repo)))
        if filters.language is not None:
            must.append(
                qm.FieldCondition(key="language", match=qm.MatchValue(value=filters.language))
            )
        if filters.source_type is not None:
            must.append(
                qm.FieldCondition(key="source_type", match=qm.MatchValue(value=filters.source_type))
            )
        if filters.tags:
            must.append(qm.FieldCondition(key="tags", match=qm.MatchAny(any=filters.tags)))
        if filters.created_after is not None or filters.created_before is not None:
            must.append(
                qm.FieldCondition(
                    key="created_at",
                    range=qm.Range(
                        gte=_epoch(filters.created_after),
                        lte=_epoch(filters.created_before),
                    ),
                )
            )
        if not filters.include_expired:
            valid_at = _epoch(filters.valid_at or utcnow())
            should.extend(
                [
                    qm.IsEmptyCondition(is_empty=qm.PayloadField(key="valid_to")),
                    qm.FieldCondition(key="valid_to", range=qm.Range(gte=valid_at)),
                ]
            )
    else:
        valid_at = _epoch(utcnow())
        should.extend(
            [
                qm.IsEmptyCondition(is_empty=qm.PayloadField(key="valid_to")),
                qm.FieldCondition(key="valid_to", range=qm.Range(gte=valid_at)),
            ]
        )
    return qm.Filter(must=must or None, should=should or None)


class QdrantVectorIndex:
    """VectorIndex over embedded Qdrant."""

    def __init__(self, path: Path | str, collection: str) -> None:
        self._path = str(path)
        self._collection = collection
        self._client: QdrantClient | None = None

    async def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantVectorIndex is not initialized; call initialize() first")
        return self._client

    async def initialize(self, dimension: int) -> None:
        if self._client is not None:
            return

        def _init() -> QdrantClient:
            client = QdrantClient(path=self._path)
            if not client.collection_exists(self._collection):
                client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
                )
                # Payload indexes are a no-op in embedded local mode; create
                # them only when talking to a server.
                if client._client.__class__.__name__ != "QdrantLocal":
                    for field, schema in _PAYLOAD_INDEXES.items():
                        client.create_payload_index(
                            collection_name=self._collection,
                            field_name=field,
                            field_schema=schema,
                        )
            return client

        self._client = await anyio.to_thread.run_sync(_init)

    async def close(self) -> None:
        if self._client is not None:
            await self._run(self._client.close)
            self._client = None

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        structs = [qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points]
        await self._run(self.client.upsert, collection_name=self._collection, points=structs)

    async def delete(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        await self._run(
            self.client.delete,
            collection_name=self._collection,
            points_selector=qm.PointIdsList(points=list(point_ids)),
        )

    async def search(
        self,
        vector: Sequence[float],
        filters: SearchFilters | None,
        limit: int,
        item_kinds: Sequence[str] | None = None,
    ) -> list[ScoredId]:
        query_filter = _filters_to_qdrant(filters, item_kinds)
        needs_post_filter = filters is not None and filters.path_prefix is not None
        fetch = limit * 2 if needs_post_filter else limit
        response = await self._run(
            self.client.query_points,
            collection_name=self._collection,
            query=list(vector),
            query_filter=query_filter,
            limit=fetch,
            with_payload=True,
        )
        results: list[ScoredId] = []
        for point in response.points:
            payload = point.payload or {}
            if needs_post_filter:
                path = payload.get("path")
                assert filters is not None
                if not (path or "").startswith(filters.path_prefix or ""):
                    continue
            item_id = payload.get("item_id")
            if item_id is None:
                continue
            results.append(ScoredId(id=item_id, score=point.score))
            if len(results) >= limit:
                break
        return results
