"""Deterministic retrieval and operational metrics for saved S4 results."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence


def retrieval_metrics(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    cutoffs: Sequence[int],
    mrr_cutoff: int,
    ndcg_cutoff: int,
) -> dict[str, float]:
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")
    if len(set(retrieved_chunk_ids)) != len(retrieved_chunk_ids):
        raise ValueError("retrieved_chunk_ids must not contain duplicate IDs")
    if any(type(value) is not int or value <= 0 for value in (*cutoffs, mrr_cutoff, ndcg_cutoff)):
        raise ValueError("metric cutoffs must be positive integers")
    relevant = set(relevant_chunk_ids)
    result: dict[str, float] = {}
    for cutoff in cutoffs:
        result[f"recall_at_{cutoff}"] = float(
            any(chunk_id in relevant for chunk_id in retrieved_chunk_ids[:cutoff])
        )
    first_rank = next(
        (rank for rank, chunk_id in enumerate(retrieved_chunk_ids[:mrr_cutoff], 1) if chunk_id in relevant),
        None,
    )
    result[f"mrr_at_{mrr_cutoff}"] = 0.0 if first_rank is None else 1.0 / first_rank
    gains = [
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_chunk_ids[:ndcg_cutoff], 1)
        if chunk_id in relevant
    ]
    ideal_count = min(len(relevant), ndcg_cutoff)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    result[f"ndcg_at_{ndcg_cutoff}"] = sum(gains) / ideal
    return result


def aggregate_retrieval_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    count = 0
    for row in rows:
        if row.get("status") != "success" or row.get("answerable") is not True:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise TypeError("successful answerable rows require metrics")
        count += 1
        for name, value in metrics.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("metric values must be finite")
            values[str(name)].append(numeric)
    result: dict[str, object] = {"evaluated_answerable_count": count}
    for name in sorted(values):
        result[name] = sum(values[name]) / len(values[name])
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_operations(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    latencies: defaultdict[str, list[float]] = defaultdict(list)
    usage = {"input_tokens": 0, "output_tokens": 0, "api_cost_usd": 0.0}
    success_count = 0
    failed_count = 0
    for row in rows:
        status = row.get("status")
        if status == "success":
            success_count += 1
        elif status == "failed":
            failed_count += 1
        else:
            raise ValueError("row status must be success or failed")
        latency = row.get("latency_ms")
        row_usage = row.get("usage")
        if not isinstance(latency, Mapping) or not isinstance(row_usage, Mapping):
            raise TypeError("rows require latency_ms and usage objects")
        for stage, value in latency.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("latency values must be finite and non-negative")
            latencies[str(stage)].append(numeric)
        for key in ("input_tokens", "output_tokens"):
            value = row_usage.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
            usage[key] += value
        cost = float(row_usage.get("api_cost_usd", -1))
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("api_cost_usd must be finite and non-negative")
        usage["api_cost_usd"] += cost
    return {
        "success_count": success_count,
        "failed_count": failed_count,
        "latency_ms": {
            stage: {"p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95)}
            for stage, values in sorted(latencies.items())
        },
        "usage": usage,
    }

