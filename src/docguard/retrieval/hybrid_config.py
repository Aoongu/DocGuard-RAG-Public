"""Strict immutable configuration for S5 BM25 and RRF retrieval."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "retrieval_version",
        "tokenizer_version",
        "bm25_enabled",
        "dense_enabled",
        "bm25",
        "rrf",
        "chunking_version",
        "chunks_sha256",
    }
)
_BM25_FIELDS = frozenset({"k1", "b"})
_RRF_FIELDS = frozenset(
    {
        "k_candidates",
        "dense_depth_candidates",
        "bm25_depth_candidates",
        "selection_metrics",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _strict_fields(data: dict[str, Any], expected: frozenset[str], label: str) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _int_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or not all(type(item) is int for item in value):
        raise ValueError(f"{label} must be a non-empty integer list")
    return tuple(value)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a non-empty string list")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class HybridConfig:
    schema_version: int
    retrieval_version: str
    tokenizer_version: str
    bm25_enabled: bool
    dense_enabled: bool
    k1: float
    b: float
    rrf_k_candidates: tuple[int, ...]
    dense_depth_candidates: tuple[int, ...]
    bm25_depth_candidates: tuple[int, ...]
    selection_metrics: tuple[str, ...]
    chunking_version: str
    chunks_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> HybridConfig:
        data = _mapping(value, "hybrid config")
        _strict_fields(data, _TOP_FIELDS, "hybrid config")
        bm25 = _mapping(data["bm25"], "bm25")
        rrf = _mapping(data["rrf"], "rrf")
        _strict_fields(bm25, _BM25_FIELDS, "bm25")
        _strict_fields(rrf, _RRF_FIELDS, "rrf")
        config = cls(
            schema_version=data["schema_version"],
            retrieval_version=data["retrieval_version"],
            tokenizer_version=data["tokenizer_version"],
            bm25_enabled=data["bm25_enabled"],
            dense_enabled=data["dense_enabled"],
            k1=_finite_float(bm25["k1"], "bm25.k1"),
            b=_finite_float(bm25["b"], "bm25.b"),
            rrf_k_candidates=_int_tuple(rrf["k_candidates"], "rrf.k_candidates"),
            dense_depth_candidates=_int_tuple(
                rrf["dense_depth_candidates"], "rrf.dense_depth_candidates"
            ),
            bm25_depth_candidates=_int_tuple(
                rrf["bm25_depth_candidates"], "rrf.bm25_depth_candidates"
            ),
            selection_metrics=_string_tuple(
                rrf["selection_metrics"], "rrf.selection_metrics"
            ),
            chunking_version=data["chunking_version"],
            chunks_sha256=data["chunks_sha256"],
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.retrieval_version != "s5-hybrid-v1":
            raise ValueError("retrieval_version must be s5-hybrid-v1")
        if self.tokenizer_version != "s5-tech-tokenizer-v1":
            raise ValueError("tokenizer_version must be s5-tech-tokenizer-v1")
        if type(self.bm25_enabled) is not bool or not self.bm25_enabled:
            raise ValueError("bm25_enabled must be true")
        if type(self.dense_enabled) is not bool or not self.dense_enabled:
            raise ValueError("dense_enabled must be true")
        if self.k1 != 1.2 or self.b != 0.75:
            raise ValueError("BM25 constants must be k1=1.2 and b=0.75")
        if self.rrf_k_candidates != (20, 60):
            raise ValueError("rrf.k_candidates must be [20, 60]")
        if self.dense_depth_candidates != (20, 50):
            raise ValueError("rrf.dense_depth_candidates must be [20, 50]")
        if self.bm25_depth_candidates != (20, 50):
            raise ValueError("rrf.bm25_depth_candidates must be [20, 50]")
        if self.selection_metrics != (
            "ndcg_at_10",
            "mrr_at_10",
            "recall_at_10",
            "recall_at_5",
        ):
            raise ValueError("rrf.selection_metrics do not match the frozen order")
        if self.chunking_version != "s2-chunk-v1":
            raise ValueError("chunking_version must be s2-chunk-v1")
        if not isinstance(self.chunks_sha256, str) or _SHA256_RE.fullmatch(
            self.chunks_sha256
        ) is None:
            raise ValueError("chunks_sha256 must be a lowercase SHA-256")


def load_hybrid_config(path: Path) -> HybridConfig:
    with path.open("r", encoding="utf-8") as file:
        return HybridConfig.from_dict(json.load(file))
