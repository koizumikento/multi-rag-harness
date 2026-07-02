"""Local model loading utilities."""

from __future__ import annotations

from multi_rag_harness.config import Settings
from multi_rag_harness.models.embedding import E5Embedder, EmbeddingModel
from multi_rag_harness.models.reranker import CrossEncoderReranker, Reranker


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
    return E5Embedder(
        model_name=settings.embedding.model,
        device=resolve_device(settings.embedding.device),
        dimension=settings.embedding.dimension,
        batch_size=settings.embedding.batch_size,
    )


def create_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker(
        model_name=settings.reranker.model,
        device=resolve_device(settings.reranker.device),
    )
