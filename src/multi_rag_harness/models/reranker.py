"""Reranker model adapters.

Cross-encoder loading is lazy: nothing downloads until the first score call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import anyio.to_thread

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
