"""Paired bootstrap comparison for frozen per-question metrics."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    seed: int,
    samples: int,
    confidence_level: float,
) -> dict[str, int | float]:
    if not baseline:
        raise ValueError("paired inputs must not be empty")
    if set(baseline) != set(candidate):
        raise ValueError("paired inputs must contain the same question IDs")
    if type(seed) is not int or type(samples) is not int or samples <= 0:
        raise ValueError("seed and samples must be valid integers")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    question_ids = sorted(baseline)
    differences = []
    for question_id in question_ids:
        difference = float(candidate[question_id]) - float(baseline[question_id])
        if not math.isfinite(difference):
            raise ValueError("paired metric values must be finite")
        differences.append(difference)
    rng = random.Random(seed)
    count = len(differences)
    bootstrap_means = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    tail = (1.0 - confidence_level) / 2.0
    return {
        "paired_question_count": count,
        "seed": seed,
        "samples": samples,
        "confidence_level": confidence_level,
        "mean_difference": sum(differences) / count,
        "ci_lower": _percentile(bootstrap_means, tail),
        "ci_upper": _percentile(bootstrap_means, 1.0 - tail),
    }
