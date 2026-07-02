"""Embedding model adapters.

Two providers: local sentence-transformers (default) and an OpenAI-compatible
``/embeddings`` API (OpenAI, TEI, vLLM, Ollama, ...). E5-style models require
``query: `` / ``passage: `` prefixes; the prefix handling is a pure function
so it can be unit tested without model weights. Local model loading is lazy:
nothing downloads until the first embed call. API models receive raw text
without E5 prefixes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import anyio.to_thread
import httpx

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

E5_PREFIXES = {"query": "query: ", "passage": "passage: "}


def apply_e5_prefix(texts: Sequence[str], mode: Literal["query", "passage"]) -> list[str]:
    prefix = E5_PREFIXES[mode]
    return [prefix + text for text in texts]


@runtime_checkable
class EmbeddingModel(Protocol):
    @property
    def dimension(self) -> int: ...
    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


class E5Embedder:
    """Lazy sentence-transformers embedder with E5 prefix handling."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dimension: int = 768,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._dimension = dimension
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = apply_e5_prefix(texts, "query")
        return await anyio.to_thread.run_sync(self._encode, prefixed)

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = apply_e5_prefix(texts, "passage")
        return await anyio.to_thread.run_sync(self._encode, prefixed)


class OpenAICompatEmbedder:
    """Embedder over an OpenAI-compatible ``POST {base_url}/embeddings``
    endpoint. ``base_url`` includes the version prefix (e.g.
    ``https://api.openai.com/v1``). The api key is optional for self-hosted
    servers. No E5 prefixes are applied."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimension: int,
        api_key: str | None = None,
        batch_size: int = 32,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._api_key = api_key
        self._batch_size = batch_size
        self._timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

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

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = await client.post("/embeddings", json={"model": self._model, "input": batch})
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            batch_vectors = [item["embedding"] for item in data]
            for vector in batch_vectors:
                if len(vector) != self._dimension:
                    raise ValueError(
                        f"embedding endpoint returned {len(vector)}-dimensional vectors "
                        f"but embedding.dimension is {self._dimension}; align the config "
                        f"with the model '{self._model}'"
                    )
            vectors.extend(batch_vectors)
        return vectors

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts)
