"""Unit tests for E5 prefix handling and lazy model loading."""

from typing import Any, cast

from multi_rag_harness.models.embedding import E5Embedder, apply_e5_prefix


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int, bool, bool, bool]] = []

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[FakeVector]:
        self.calls.append(
            (texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar)
        )
        return [FakeVector([float(index), 1.0]) for index, _ in enumerate(texts)]


def test_apply_e5_prefix_query() -> None:
    assert apply_e5_prefix(["how to test"], "query") == ["query: how to test"]


def test_apply_e5_prefix_passage() -> None:
    assert apply_e5_prefix(["a doc", "b doc"], "passage") == [
        "passage: a doc",
        "passage: b doc",
    ]


def test_e5_embedder_is_lazy() -> None:
    embedder = E5Embedder("intfloat/multilingual-e5-base", device="cpu")
    assert embedder._model is None
    assert embedder.dimension == 768


async def test_e5_embedder_empty_input_does_not_load_model() -> None:
    embedder = E5Embedder("intfloat/multilingual-e5-base", device="cpu")
    assert await embedder.embed_queries([]) == []
    assert await embedder.embed_passages([]) == []
    assert embedder._model is None


async def test_e5_embedder_encodes_prefixed_text_with_configured_batch_size() -> None:
    model = FakeSentenceTransformer()
    embedder = E5Embedder("intfloat/multilingual-e5-base", device="cpu", batch_size=7)
    cast(Any, embedder)._model = model

    assert await embedder.embed_queries(["what"]) == [[0.0, 1.0]]
    assert await embedder.embed_passages(["doc"]) == [[0.0, 1.0]]
    assert model.calls == [
        (["query: what"], 7, True, True, False),
        (["passage: doc"], 7, True, True, False),
    ]
