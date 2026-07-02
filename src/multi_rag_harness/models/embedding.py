"""Embedding model adapters.

E5-style models require ``query: `` / ``passage: `` prefixes; the prefix
handling is a pure function so it can be unit tested without model weights.
Model loading is lazy: nothing downloads until the first embed call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import anyio.to_thread

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
