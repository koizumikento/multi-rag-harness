"""Unit tests for API-backed model adapters and the provider factories."""

import json

import httpx
import pytest

from multi_rag_harness.config import Settings
from multi_rag_harness.models.embedding import E5Embedder, OpenAICompatEmbedder
from multi_rag_harness.models.local import create_embedder, create_reranker
from multi_rag_harness.models.reranker import ApiReranker, CrossEncoderReranker

DIM = 4


def embedding_transport(captured: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        # Return items out of order to verify index-based sorting.
        data = [
            {"index": i, "embedding": [float(i)] * DIM} for i in reversed(range(len(body["input"])))
        ]
        return httpx.Response(200, json={"data": data})

    return httpx.MockTransport(handler)


async def test_openai_compat_embedder_requests_and_sorting() -> None:
    captured: list[httpx.Request] = []
    embedder = OpenAICompatEmbedder(
        base_url="https://api.example.com/v1",
        model="text-embedding-3-small",
        dimension=DIM,
        api_key="sk-test",
        transport=embedding_transport(captured),
    )
    vectors = await embedder.embed_queries(["first", "second"])
    assert vectors == [[0.0] * DIM, [1.0] * DIM]  # sorted by index

    request = captured[0]
    assert request.url.path == "/v1/embeddings"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == ["first", "second"]  # raw text, no E5 prefixes
    await embedder.close()


async def test_openai_compat_embedder_batches() -> None:
    captured: list[httpx.Request] = []
    embedder = OpenAICompatEmbedder(
        base_url="https://api.example.com/v1",
        model="m",
        dimension=DIM,
        batch_size=2,
        transport=embedding_transport(captured),
    )
    vectors = await embedder.embed_passages(["a", "b", "c"])
    assert len(vectors) == 3
    assert len(captured) == 2  # 2 + 1
    # No api key -> no auth header.
    assert "authorization" not in captured[0].headers
    await embedder.close()


async def test_openai_compat_embedder_dimension_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    embedder = OpenAICompatEmbedder(
        base_url="https://api.example.com/v1",
        model="m",
        dimension=DIM,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError, match="dimension"):
        await embedder.embed_queries(["text"])
    await embedder.close()


async def test_api_reranker_cohere_style_response() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    reranker = ApiReranker(
        base_url="https://api.example.com/v1",
        model="rerank-1",
        api_key="key",
        transport=httpx.MockTransport(handler),
    )
    scores = await reranker.score("query", ["p0", "p1"])
    assert scores == [0.2, 0.9]

    request = captured[0]
    assert request.url.path == "/v1/rerank"
    body = json.loads(request.content)
    assert body == {"model": "rerank-1", "query": "query", "documents": ["p0", "p1"], "top_n": 2}
    await reranker.close()


async def test_api_reranker_bare_list_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"index": 0, "score": 0.7}])

    reranker = ApiReranker(
        base_url="https://api.example.com",
        model="m",
        transport=httpx.MockTransport(handler),
    )
    assert await reranker.score("q", ["only"]) == [0.7]
    assert await reranker.score("q", []) == []
    await reranker.close()


def test_factory_provider_switch() -> None:
    local = Settings()
    assert isinstance(create_embedder(local), E5Embedder)
    assert isinstance(create_reranker(local), CrossEncoderReranker)

    api = Settings()
    api.embedding.provider = "api"
    api.embedding.base_url = "https://api.example.com/v1"
    api.reranker.provider = "api"
    api.reranker.base_url = "https://api.example.com/v1"
    assert isinstance(create_embedder(api), OpenAICompatEmbedder)
    assert isinstance(create_reranker(api), ApiReranker)


def test_factory_requires_base_url_for_api_provider() -> None:
    settings = Settings()
    settings.embedding.provider = "api"
    with pytest.raises(ValueError, match="MRH_EMBEDDING__BASE_URL"):
        create_embedder(settings)

    settings2 = Settings()
    settings2.reranker.provider = "api"
    with pytest.raises(ValueError, match="MRH_RERANKER__BASE_URL"):
        create_reranker(settings2)
