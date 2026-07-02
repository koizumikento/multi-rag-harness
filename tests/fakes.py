"""Deterministic fakes for tests. No model downloads, no external calls."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence


class FakeEmbedder:
    """Deterministic bag-of-token-hashes embedder.

    Texts sharing tokens get similar vectors, so hybrid/vector retrieval tests
    behave sensibly. Records every text it receives so callers can assert on
    what was embedded.
    """

    def __init__(self, dimension: int = 32) -> None:
        self._dimension = dimension
        self.query_texts: list[str] = []
        self.passage_texts: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self._dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            values[int(digest, 16) % self._dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.query_texts.extend(texts)
        return [self._vector(text) for text in texts]

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.passage_texts.extend(texts)
        return [self._vector(text) for text in texts]


class FakeReranker:
    """Scores passages by token-overlap ratio with the query."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        query_tokens = set(query.lower().split())
        scores = []
        for passage in passages:
            passage_tokens = set(passage.lower().split())
            if not query_tokens or not passage_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & passage_tokens)
            scores.append(overlap / len(query_tokens))
        return scores


class FakeCodexClient:
    """Returns canned extraction responses in order; records prompts."""

    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    async def run_structured(self, prompt: str, output_schema: dict) -> tuple[str, str | None]:
        self.prompts.append(prompt)
        self.schemas.append(output_schema)
        if not self._responses:
            raise RuntimeError("FakeCodexClient has no responses left")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, f"fake-thread-{len(self.prompts)}"
