"""Reranker model adapters.

Two providers: local cross-encoder (default; lazy loading, nothing downloads
until the first score call) and a Cohere/Jina-compatible ``/rerank`` API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import anyio.to_thread
import httpx

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@runtime_checkable
class Reranker(Protocol):
    async def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Lazy sentence-transformers cross-encoder reranker."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load()
        scores = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        return [float(score) for score in scores]

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        return await anyio.to_thread.run_sync(self._predict, pairs)


class ApiReranker:
    """Reranker over a Cohere/Jina-compatible ``POST {base_url}/rerank``
    endpoint. Request: ``{model, query, documents, top_n}``; response:
    ``{"results": [{"index": i, "relevance_score": s}]}`` (a bare list with
    ``score`` keys is also accepted)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
                transport=self._transport,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        client = self._get_client()
        response = await client.post(
            "/rerank",
            json={
                "model": self._model,
                "query": query,
                "documents": list(passages),
                "top_n": len(passages),
            },
        )
        response.raise_for_status()
        payload: Any = response.json()
        items = payload.get("results", []) if isinstance(payload, dict) else payload
        scores = [0.0] * len(passages)
        for item in items:
            index = item["index"]
            if not 0 <= index < len(passages):
                continue
            value = item.get("relevance_score", item.get("score", 0.0))
            scores[index] = float(value)
        return scores
