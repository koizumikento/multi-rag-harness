"""Model factories: local (default) or external API, selected by config."""

from __future__ import annotations

from multi_rag_harness.config import Settings
from multi_rag_harness.models.embedding import E5Embedder, EmbeddingModel, OpenAICompatEmbedder
from multi_rag_harness.models.reranker import ApiReranker, CrossEncoderReranker, Reranker


def resolve_device(spec: str) -> str:
    """Resolve a device spec; ``auto`` probes cuda, then mps, then cpu."""
    if spec != "auto":
        return spec
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_embedder(settings: Settings) -> EmbeddingModel:
    if settings.embedding.provider == "api":
        if not settings.embedding.base_url:
            raise ValueError(
                "embedding.base_url is required when embedding.provider='api' "
                "(set MRH_EMBEDDING__BASE_URL)"
            )
        return OpenAICompatEmbedder(
            base_url=settings.embedding.base_url,
            model=settings.embedding.model,
            dimension=settings.embedding.dimension,
            api_key=settings.embedding.api_key,
            batch_size=settings.embedding.batch_size,
        )
    return E5Embedder(
        model_name=settings.embedding.model,
        device=resolve_device(settings.embedding.device),
        dimension=settings.embedding.dimension,
        batch_size=settings.embedding.batch_size,
    )


def create_reranker(settings: Settings) -> Reranker:
    if settings.reranker.provider == "api":
        if not settings.reranker.base_url:
            raise ValueError(
                "reranker.base_url is required when reranker.provider='api' "
                "(set MRH_RERANKER__BASE_URL)"
            )
        return ApiReranker(
            base_url=settings.reranker.base_url,
            model=settings.reranker.model,
            api_key=settings.reranker.api_key,
        )
    return CrossEncoderReranker(
        model_name=settings.reranker.model,
        device=resolve_device(settings.reranker.device),
    )
