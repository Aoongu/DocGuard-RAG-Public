"""Deterministic citation and boundary-generation metrics for S7."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence


def citation_metrics(
    cited_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    known_chunk_ids: Sequence[str],
) -> dict[str, float]:
    """Score citations after validating both predictions and gold IDs."""
    cited = tuple(cited_chunk_ids)
    relevant = tuple(relevant_chunk_ids)
    known = set(known_chunk_ids)
    if len(set(cited)) != len(cited):
        raise ValueError("duplicate cited chunk IDs")
    if len(set(relevant)) != len(relevant):
        raise ValueError("duplicate relevant chunk IDs")
    if not set(relevant).issubset(known):
        raise ValueError("unknown gold chunk ID")
    if not set(cited).issubset(known):
        raise ValueError("unknown cited chunk ID")
    overlap = len(set(cited) & set(relevant))
    return {
        "citation_precision": 0.0 if not cited else overlap / len(cited),
        "citation_recall": 0.0 if not relevant else overlap / len(relevant),
    }


def aggregate_generation_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Keep answerable citation quality separate from boundary behavior."""
    citation_values: defaultdict[str, list[float]] = defaultdict(list)
    answerable_count = 0
    boundary_count = 0
    boundary_answers = 0
    boundary_citations = 0
    for row in rows:
        if row.get("status") != "success":
            continue
        if row.get("answerable") is True:
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                raise TypeError("successful answerable rows require metrics")
            answerable_count += 1
            for name in ("citation_precision", "citation_recall"):
                value = float(metrics[name])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("citation metrics must be finite values in [0, 1]")
                citation_values[name].append(value)
        elif row.get("answerable") is False:
            boundary_count += 1
            answer = row.get("answer")
            citations = row.get("citations")
            if not isinstance(answer, str) or not isinstance(citations, list):
                raise TypeError("successful boundary rows require answer and citations")
            boundary_answers += int(bool(answer.strip()))
            boundary_citations += int(bool(citations))
        else:
            raise TypeError("answerable must be boolean")
    answerable: dict[str, object] = {"evaluated_count": answerable_count}
    for name in ("citation_precision", "citation_recall"):
        values = citation_values[name]
        answerable[name] = 0.0 if not values else sum(values) / len(values)
    boundary = {
        "evaluated_count": boundary_count,
        "answer_rate": 0.0 if not boundary_count else boundary_answers / boundary_count,
        "citation_rate": 0.0 if not boundary_count else boundary_citations / boundary_count,
    }
    return {"answerable": answerable, "boundary": boundary}
