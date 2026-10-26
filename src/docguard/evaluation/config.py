"""Strict loader for the frozen S4 evaluation configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FIELDS = frozenset(
    {
        "schema_version",
        "evaluation_version",
        "dataset_version",
        "allowed_splits",
        "retrieval_cutoffs",
        "mrr_cutoff",
        "ndcg_cutoff",
        "retrieval_top_k",
        "bootstrap_seed",
        "bootstrap_samples",
        "confidence_level",
        "chunking_version",
        "corpus_manifest_sha256",
        "chunks_sha256",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _strict_fields(data: dict[str, Any]) -> None:
    unknown = set(data) - _FIELDS
    missing = _FIELDS - set(data)
    if unknown:
        raise ValueError(f"evaluation config contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"evaluation config is missing fields: {sorted(missing)}")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    schema_version: int
    evaluation_version: str
    dataset_version: str
    allowed_splits: tuple[str, ...]
    retrieval_cutoffs: tuple[int, ...]
    mrr_cutoff: int
    ndcg_cutoff: int
    retrieval_top_k: int
    bootstrap_seed: int
    bootstrap_samples: int
    confidence_level: float
    chunking_version: str
    corpus_manifest_sha256: str
    chunks_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.evaluation_version != "s4-eval-v1" or self.dataset_version != "s4-golden-v1":
            raise ValueError("evaluation and dataset versions must match S4 v1")
        if self.allowed_splits != ("development", "validation"):
            raise ValueError("allowed_splits must exclude test and preserve frozen order")
        if self.retrieval_cutoffs != (1, 5, 10):
            raise ValueError("retrieval_cutoffs must be [1, 5, 10]")
        if self.mrr_cutoff != 10 or self.ndcg_cutoff != 10 or self.retrieval_top_k != 10:
            raise ValueError("S4 retrieval cutoffs must be 10")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be a non-negative integer")
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 1000:
            raise ValueError("bootstrap_samples must be at least 1000")
        if self.confidence_level != 0.95:
            raise ValueError("confidence_level must be 0.95")
        if self.chunking_version != "s2-chunk-v1":
            raise ValueError("chunking_version must be s2-chunk-v1")
        for name, value in (
            ("corpus_manifest_sha256", self.corpus_manifest_sha256),
            ("chunks_sha256", self.chunks_sha256),
        ):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")


def load_evaluation_config(path: Path) -> EvaluationConfig:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("evaluation config must be an object")
    _strict_fields(data)
    allowed = data["allowed_splits"]
    cutoffs = data["retrieval_cutoffs"]
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ValueError("allowed_splits must be a string list")
    if not isinstance(cutoffs, list) or not all(type(item) is int for item in cutoffs):
        raise ValueError("retrieval_cutoffs must be an integer list")
    return EvaluationConfig(
        schema_version=data["schema_version"],
        evaluation_version=data["evaluation_version"],
        dataset_version=data["dataset_version"],
        allowed_splits=tuple(allowed),
        retrieval_cutoffs=tuple(cutoffs),
        mrr_cutoff=data["mrr_cutoff"],
        ndcg_cutoff=data["ndcg_cutoff"],
        retrieval_top_k=data["retrieval_top_k"],
        bootstrap_seed=data["bootstrap_seed"],
        bootstrap_samples=data["bootstrap_samples"],
        confidence_level=data["confidence_level"],
        chunking_version=data["chunking_version"],
        corpus_manifest_sha256=data["corpus_manifest_sha256"],
        chunks_sha256=data["chunks_sha256"],
    )

