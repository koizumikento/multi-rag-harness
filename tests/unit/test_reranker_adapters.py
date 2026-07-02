"""Unit tests for local reranker adapter behavior without model downloads."""

from typing import Any, cast

from multi_rag_harness.models.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], bool, bool]] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[float]:
        self.calls.append((pairs, convert_to_numpy, show_progress_bar))
        return [0.25, 0.75]


async def test_cross_encoder_reranker_empty_passages_does_not_load_model() -> None:
    reranker = CrossEncoderReranker("reranker", device="cpu")
    assert await reranker.score("query", []) == []
    assert reranker._model is None


async def test_cross_encoder_reranker_scores_query_passage_pairs() -> None:
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker("reranker", device="cpu")
    cast(Any, reranker)._model = model

    scores = await reranker.score("q", ["p0", "p1"])
    assert scores == [0.25, 0.75]
    assert model.calls == [([("q", "p0"), ("q", "p1")], True, False)]
