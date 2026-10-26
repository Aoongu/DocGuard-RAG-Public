"""Strict loader for the frozen S8 refusal contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

_FIELDS = {
    "schema_version",
    "refusal_version",
    "supported_versions",
    "threshold_grid",
    "selection_objective",
    "token_pattern",
    "stopwords",
    "domain_terms",
    "version_conflict",
    "input_sha256",
    "test_execution_status",
}


@dataclass(frozen=True, slots=True)
class ThresholdGrid:
    top1: tuple[float, ...]
    top3_mean: tuple[float, ...]
    coverage: tuple[float, ...]
    margin: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VersionConflictConfig:
    min_score: float
    max_token_jaccard: float


@dataclass(frozen=True, slots=True)
class RefusalConfig:
    schema_version: int
    refusal_version: str
    supported_versions: tuple[str, ...]
    threshold_grid: ThresholdGrid
    selection_objective: tuple[str, ...]
    token_pattern: str
    stopwords: tuple[str, ...]
    domain_terms: tuple[str, ...]
    version_conflict: VersionConflictConfig
    input_sha256: tuple[tuple[str, str], ...]
    test_execution_status: str


def _numbers(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique finite numbers")
    return result


def load_refusal_config(path: Path) -> RefusalConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("refusal config fields do not match the frozen schema")
    if payload["schema_version"] != 1 or payload["refusal_version"] != "s8-refusal-v1":
        raise ValueError("unsupported refusal config identity")
    if payload["supported_versions"] != ["17.11", "18.6"]:
        raise ValueError("supported_versions must be the frozen corpus versions")
    grid = payload["threshold_grid"]
    if not isinstance(grid, dict) or set(grid) != {"top1", "top3_mean", "coverage", "margin"}:
        raise ValueError("threshold_grid fields do not match the frozen schema")
    objectives = tuple(payload["selection_objective"])
    expected_objectives = (
        "macro_f1",
        "false_accept_count",
        "refusal_precision",
        "answerable_coverage",
        "threshold_lexicographic",
    )
    if objectives != expected_objectives:
        raise ValueError("selection_objective differs from the preregistered order")
    stopwords = tuple(payload["stopwords"])
    domain_terms = tuple(payload["domain_terms"])
    if not stopwords or len(set(stopwords)) != len(stopwords):
        raise ValueError("stopwords must be non-empty and unique")
    if not domain_terms or len(set(domain_terms)) != len(domain_terms):
        raise ValueError("domain_terms must be non-empty and unique")
    conflict = payload["version_conflict"]
    if not isinstance(conflict, dict) or set(conflict) != {"min_score", "max_token_jaccard"}:
        raise ValueError("version_conflict fields do not match the frozen schema")
    inputs = payload["input_sha256"]
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("input_sha256 must be a non-empty object")
    for name, digest in inputs.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("input_sha256 contains an invalid identity")
    if payload["test_execution_status"] != "locked_until_s12":
        raise ValueError("Test must remain locked until S12")
    return RefusalConfig(
        schema_version=1,
        refusal_version="s8-refusal-v1",
        supported_versions=("17.11", "18.6"),
        threshold_grid=ThresholdGrid(
            top1=_numbers(grid["top1"], "top1"),
            top3_mean=_numbers(grid["top3_mean"], "top3_mean"),
            coverage=_numbers(grid["coverage"], "coverage"),
            margin=_numbers(grid["margin"], "margin"),
        ),
        selection_objective=objectives,
        token_pattern=str(payload["token_pattern"]),
        stopwords=stopwords,
        domain_terms=domain_terms,
        version_conflict=VersionConflictConfig(
            min_score=float(conflict["min_score"]),
            max_token_jaccard=float(conflict["max_token_jaccard"]),
        ),
        input_sha256=tuple(sorted(inputs.items())),
        test_execution_status="locked_until_s12",
    )
