"""Strict S11 engineering configuration and compatibility checks."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_IDENTITIES = {
    "api_version": "s9-api-v1",
    "chunking_version": "s2-chunk-v1",
    "dense_index_version": "s3-dense-v1",
    "hybrid_version": "s5-hybrid-v1",
    "reranker_version": "s6-reranker-v1",
    "generation_version": "s7-deepseek-flash-v1",
    "refusal_version": "s8-refusal-v1",
    "test_execution_status": "locked_until_s12",
}
_FIELDS = {
    "schema_version",
    "engineering_version",
    *_IDENTITIES,
    "score_tolerance",
    "regression",
}
_REGRESSION_FIELDS = {"min_recall_at_5", "min_mrr_at_10", "max_failed_queries"}


@dataclass(frozen=True, slots=True)
class RegressionThresholds:
    min_recall_at_5: float
    min_mrr_at_10: float
    max_failed_queries: int


@dataclass(frozen=True, slots=True)
class EngineeringConfig:
    schema_version: int
    engineering_version: str
    api_version: str
    chunking_version: str
    dense_index_version: str
    hybrid_version: str
    reranker_version: str
    generation_version: str
    refusal_version: str
    test_execution_status: str
    score_tolerance: float
    regression: RegressionThresholds


def _ratio(value: object, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite ratio from 0 to 1")
    return float(value)


def load_engineering_config(path: Path) -> EngineeringConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("S11 engineering config fields do not match the frozen schema")
    if payload["schema_version"] != 1 or payload["engineering_version"] != "s11-engineering-v1":
        raise ValueError("unsupported S11 engineering config identity")
    for name, expected in _IDENTITIES.items():
        if payload[name] != expected:
            raise ValueError(f"{name} must remain {expected}")
    tolerance = payload["score_tolerance"]
    if type(tolerance) not in {int, float} or not math.isfinite(tolerance) or not 0 < tolerance <= 1e-3:
        raise ValueError("score_tolerance must be finite and in (0, 1e-3]")
    regression = payload["regression"]
    if not isinstance(regression, dict) or set(regression) != _REGRESSION_FIELDS:
        raise ValueError("regression fields do not match the frozen schema")
    failures = regression["max_failed_queries"]
    if type(failures) is not int or failures < 0:
        raise ValueError("max_failed_queries must be a non-negative integer")
    return EngineeringConfig(
        schema_version=1,
        engineering_version="s11-engineering-v1",
        **{name: payload[name] for name in _IDENTITIES},
        score_tolerance=float(tolerance),
        regression=RegressionThresholds(
            min_recall_at_5=_ratio(regression["min_recall_at_5"], "min_recall_at_5"),
            min_mrr_at_10=_ratio(regression["min_mrr_at_10"], "min_mrr_at_10"),
            max_failed_queries=failures,
        ),
    )


def verify_compatibility(
    config: EngineeringConfig, observed: Mapping[str, object]
) -> None:
    expected = {name: getattr(config, name) for name in _IDENTITIES}
    if set(observed) != set(expected):
        raise ValueError("compatibility identity fields are incomplete or unknown")
    for name, value in expected.items():
        if observed[name] != value:
            raise ValueError(
                f"incompatible {name}: expected {value!r}, observed {observed[name]!r}"
            )


def verify_repository_compatibility(root: Path, config: EngineeringConfig) -> None:
    def read(relative: str) -> dict[str, object]:
        value = json.loads((root / relative).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"compatibility source must be an object: {relative}")
        return value

    api = read("configs/api/s9-api-v1.json")
    chunking = read("configs/chunking/s2-chunk-v1.json")
    dense = read("configs/retrieval/s3-dense-v1.json")
    hybrid = read("configs/retrieval/s5-hybrid-v1.json")
    reranker = read("configs/retrieval/s6-reranker-v1.json")
    generation = read("configs/generation/s7-deepseek-flash-v1.json")
    refusal = read("configs/refusal/s8-refusal-v1.json")
    dataset = read("data/eval/s4-golden-v1/manifest.json")
    chunk_versions = {
        source.get("chunking_version")
        for source in (chunking, dense, hybrid, reranker, generation, dataset)
    }
    if chunk_versions != {config.chunking_version}:
        raise ValueError("chunking_version differs across frozen artifacts")
    if {
        api.get("test_execution_status"),
        refusal.get("test_execution_status"),
        dataset.get("test_execution_status"),
    } != {config.test_execution_status}:
        raise ValueError("test_execution_status differs across frozen artifacts")
    verify_compatibility(
        config,
        {
            "api_version": api.get("api_version"),
            "chunking_version": chunking.get("chunking_version"),
            "dense_index_version": dense.get("index_version"),
            "hybrid_version": hybrid.get("retrieval_version"),
            "reranker_version": reranker.get("reranking_version"),
            "generation_version": generation.get("generation_version"),
            "refusal_version": refusal.get("refusal_version"),
            "test_execution_status": dataset.get("test_execution_status"),
        },
    )
