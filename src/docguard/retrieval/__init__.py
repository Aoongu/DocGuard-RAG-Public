"""Dense retrieval primitives for DocGuard-RAG."""

from docguard.retrieval.config import DenseConfig, DenseModelSpec, load_dense_config
from docguard.retrieval.hybrid import HybridRetriever, HybridSearchHit
from docguard.retrieval.reranker import (
    CrossEncoderReranker,
    PairScore,
    RerankedSearchHit,
    RerankerError,
    rerank_hits,
)
from docguard.retrieval.reranking_pipeline import RerankingRetriever

__all__ = [
    "CrossEncoderReranker",
    "DenseConfig",
    "DenseModelSpec",
    "HybridRetriever",
    "HybridSearchHit",
    "PairScore",
    "RerankedSearchHit",
    "RerankerError",
    "RerankingRetriever",
    "load_dense_config",
    "rerank_hits",
]
