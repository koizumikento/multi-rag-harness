"""Unit tests for E5 prefix handling and lazy model loading."""

from multi_rag_harness.models.embedding import E5Embedder, apply_e5_prefix


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
