"""Unit tests for reciprocal rank fusion."""

import pytest

from multi_rag_harness.retrieval.hybrid import rrf_merge
from multi_rag_harness.storage.interfaces import ScoredId


def scored(*ids: str) -> list[ScoredId]:
    return [ScoredId(id=item_id, score=1.0) for item_id in ids]


def test_rrf_math_matches_hand_computation() -> None:
    merged = rrf_merge([scored("a", "b"), scored("b", "a")], k=60)
    by_id = {s.id: s.score for s in merged}
    assert by_id["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_prefers_items_in_both_lists() -> None:
    merged = rrf_merge([scored("both", "kw_only"), scored("both", "vec_only")])
    assert merged[0].id == "both"


def test_rrf_single_list_passthrough_order() -> None:
    merged = rrf_merge([scored("x", "y", "z")])
    assert [s.id for s in merged] == ["x", "y", "z"]


def test_rrf_deterministic_ties_by_id() -> None:
    merged = rrf_merge([scored("b"), scored("a")])
    # Both have identical scores (rank 1 in one list each): tie broken by id.
    assert [s.id for s in merged] == ["a", "b"]


def test_rrf_empty() -> None:
    assert rrf_merge([]) == []
    assert rrf_merge([[], []]) == []
