"""Deterministic regression and Qdrant parity checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from docguard.engineering.config import RegressionThresholds


@dataclass(frozen=True, slots=True)
class RankedHit:
    chunk_id: str
    score: float


def _validate_hits(hits: Sequence[RankedHit], label: str) -> None:
    ids = [hit.chunk_id for hit in hits]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains empty or duplicate chunk_id")
    if any(not math.isfinite(hit.score) for hit in hits):
        raise ValueError(f"{label} contains a non-finite score")


def compare_rankings(
    local: Sequence[RankedHit], service: Sequence[RankedHit], tolerance: float
) -> None:
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    _validate_hits(local, "local ranking")
    _validate_hits(service, "service ranking")
    local_scores = {hit.chunk_id: hit.score for hit in local}
    service_scores = {hit.chunk_id: hit.score for hit in service}
    if local_scores.keys() != service_scores.keys():
        raise AssertionError("Top-K chunk_id sets differ")
    for chunk_id, score in local_scores.items():
        if abs(score - service_scores[chunk_id]) > tolerance:
            raise AssertionError(f"score drift exceeds tolerance for {chunk_id}")

    index = 0
    while index < len(local):
        end = index + 1
        anchor = local[index].score
        while end < len(local) and abs(anchor - local[end].score) <= tolerance:
            end += 1
        if {hit.chunk_id for hit in local[index:end]} != {
            hit.chunk_id for hit in service[index:end]
        }:
            raise AssertionError(f"rank drift outside a score-tied group at rank {index + 1}")
        index = end


def assert_regression_thresholds(
    metrics: Mapping[str, object], thresholds: RegressionThresholds
) -> None:
    fields = {"recall_at_5", "mrr_at_10", "failed_queries"}
    if set(metrics) != fields:
        raise ValueError("regression metric fields are incomplete or unknown")
    recall = metrics["recall_at_5"]
    mrr = metrics["mrr_at_10"]
    failures = metrics["failed_queries"]
    if type(recall) not in {int, float} or not math.isfinite(recall):
        raise ValueError("recall_at_5 must be finite")
    if type(mrr) not in {int, float} or not math.isfinite(mrr):
        raise ValueError("mrr_at_10 must be finite")
    if type(failures) is not int or failures < 0:
        raise ValueError("failed_queries must be a non-negative integer")
    if recall < thresholds.min_recall_at_5:
        raise AssertionError("recall_at_5 is below the frozen threshold")
    if mrr < thresholds.min_mrr_at_10:
        raise AssertionError("mrr_at_10 is below the frozen threshold")
    if failures > thresholds.max_failed_queries:
        raise AssertionError("failed_queries exceeds the frozen threshold")


def evaluate_fixture(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "fixture_version",
        "questions",
    }:
        raise ValueError("regression fixture fields do not match the frozen schema")
    if payload["schema_version"] != 1 or payload["fixture_version"] != "s11-regression-v1":
        raise ValueError("unsupported regression fixture identity")
    questions = payload["questions"]
    if not isinstance(questions, list) or not questions:
        raise ValueError("regression fixture questions must be a non-empty list")
    seen: set[str] = set()
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    failures = 0
    for row in questions:
        if not isinstance(row, dict) or set(row) != {
            "query_id",
            "gold_chunk_ids",
            "retrieved_chunk_ids",
            "status",
        }:
            raise ValueError("regression question fields do not match the frozen schema")
        query_id = row["query_id"]
        gold = row["gold_chunk_ids"]
        retrieved = row["retrieved_chunk_ids"]
        if not isinstance(query_id, str) or not query_id or query_id in seen:
            raise ValueError("regression query_id must be non-empty and unique")
        if (
            not isinstance(gold, list)
            or not gold
            or not all(isinstance(item, str) and item for item in gold)
            or not isinstance(retrieved, list)
            or not all(isinstance(item, str) and item for item in retrieved)
            or len(retrieved) != len(set(retrieved))
        ):
            raise ValueError("regression chunk ID lists are invalid")
        if row["status"] not in {"success", "failed"}:
            raise ValueError("regression status must be success or failed")
        seen.add(query_id)
        if row["status"] == "failed":
            failures += 1
            recalls.append(0.0)
            reciprocal_ranks.append(0.0)
            continue
        recalls.append(float(bool(set(gold).intersection(retrieved[:5]))))
        rank = next(
            (index for index, chunk_id in enumerate(retrieved[:10], 1) if chunk_id in gold),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return {
        "recall_at_5": sum(recalls) / len(recalls),
        "mrr_at_10": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "failed_queries": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from docguard.engineering.config import load_engineering_config

    try:
        metrics = evaluate_fixture(args.fixture)
        config = load_engineering_config(args.config)
        assert_regression_thresholds(metrics, config.regression)
    except (OSError, TypeError, ValueError, AssertionError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(metrics, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
