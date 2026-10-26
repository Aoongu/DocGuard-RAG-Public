"""Read-only S12 reporting from the frozen per-question Test artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from docguard.evaluation.compare import paired_bootstrap


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_values(
    rows: Sequence[Mapping[str, object]], metric_name: str
) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        if row.get("status") != "success" or row.get("answerable") is not True:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or metric_name not in metrics:
            continue
        question_id = str(row["question_id"])
        value = float(metrics[metric_name])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {metric_name} for {question_id}")
        values[question_id] = value
    return values


def build_paired_comparison(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    metric_name: str,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, int | float | str]:
    baseline = _metric_values(baseline_rows, metric_name)
    candidate = _metric_values(candidate_rows, metric_name)
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError(f"no paired rows for {metric_name}")
    result = paired_bootstrap(
        {question_id: baseline[question_id] for question_id in shared},
        {question_id: candidate[question_id] for question_id in shared},
        seed=seed,
        samples=samples,
        confidence_level=confidence_level,
    )
    return {"metric": metric_name, **result}


def _failure_reason_codes(row: Mapping[str, object]) -> tuple[str, ...]:
    if row.get("status") != "failed":
        return ()
    error_type = str(row.get("error_type") or "")
    message = str(row.get("error_message") or "").lower()
    if error_type in {
        "CitationValidationFailure",
        "GenerationValidationError",
        "GenerationContractError",
    } or any(token in message for token in ("citation", "json schema", "answer is empty")):
        return ("generation_contract_failure",)
    if error_type in {
        "UpstreamAPIError",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
    } or any(
        token in message
        for token in ("timeout", "timed out", "429", "5xx", "response status")
    ):
        return ("upstream_failure",)
    return ("unknown",)


def _mean_metric(
    rows: Sequence[Mapping[str, object]], metric_name: str
) -> float | None:
    values = list(_metric_values(rows, metric_name).values())
    return None if not values else sum(values) / len(values)


def _latency_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        latency = row.get("latency_ms")
        if not isinstance(latency, Mapping):
            continue
        for stage, value in latency.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("latencies must be finite and non-negative")
            values[str(stage)].append(numeric)
    return {
        stage: {"p50": percentile(samples, 0.50), "p95": percentile(samples, 0.95)}
        for stage, samples in sorted(values.items())
    }


def _refusal_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    eligible = [
        row
        for row in rows
        if row.get("expected_action") in {"answer", "refuse", "clarify"}
        and row.get("action") in {"answer", "refuse", "clarify", "error"}
    ]
    actual_positive = sum(row["expected_action"] != "answer" for row in eligible)
    predicted_positive = sum(row["action"] in {"refuse", "clarify"} for row in eligible)
    true_positive = sum(
        row["expected_action"] != "answer" and row["action"] in {"refuse", "clarify"}
        for row in eligible
    )
    precision = None if predicted_positive == 0 else true_positive / predicted_positive
    recall = None if actual_positive == 0 else true_positive / actual_positive
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "n": len(eligible),
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": f1,
        "false_accept_count": sum(
            row["expected_action"] != "answer" and row["action"] == "answer"
            for row in eligible
        ),
        "false_reject_count": sum(
            row["expected_action"] == "answer" and row["action"] != "answer"
            for row in eligible
        ),
    }


def build_method_summary(
    method: str,
    rows: Sequence[Mapping[str, object]],
    *,
    fixed_total: int = 60,
) -> dict[str, object]:
    if len(rows) != fixed_total:
        raise ValueError(f"{method} requires exactly {fixed_total} rows")
    visible = [
        row
        for row in rows
        if row.get("status") == "success"
        and isinstance(row.get("answer"), str)
        and bool(str(row["answer"]).strip())
    ]
    failures = [row for row in rows if row.get("status") == "failed"]
    not_applicable = fixed_total - len(visible) - len(failures)
    if not_applicable < 0:
        raise ValueError("evaluable and failed rows must be disjoint")
    failure_codes = Counter(
        code for row in failures for code in _failure_reason_codes(row)
    )
    total_cost = 0.0
    for row in rows:
        usage = row.get("usage")
        if isinstance(usage, Mapping):
            total_cost += float(usage.get("api_cost_usd", 0.0))
    retrieval = {
        name: _mean_metric(rows, name)
        for name in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")
    }
    generation = {
        name: _mean_metric(rows, name)
        for name in ("citation_precision", "citation_recall")
    }
    return {
        "method": method,
        "fixed_total": fixed_total,
        "success_count": sum(row.get("status") == "success" for row in rows),
        "failed_count": len(failures),
        "n_evaluable": len(visible),
        "n_non_evaluable": len(failures),
        "n_not_applicable": not_applicable,
        "generation_failure_rate": len(failures) / fixed_total,
        "failure_reason_counts": dict(sorted(failure_codes.items())),
        "failure_reason_counts_are_multilabel": True,
        "retrieval": retrieval,
        "generation": generation,
        "refusal": _refusal_summary(rows),
        "latency_ms": _latency_summary(rows),
        "api_cost_usd": total_cost,
        "judge": {"status": "not_run", "failed_count": 0},
    }


_SAMPLE_CATEGORIES = (
    "success",
    "false_accept",
    "false_reject",
    "version_conflict",
    "generation_failure",
)


def _matches_category(row: Mapping[str, object], category: str) -> bool:
    if category == "success":
        return (
            row.get("status") == "success"
            and row.get("action") == "answer"
            and row.get("expected_action") == "answer"
        )
    if category == "false_accept":
        return row.get("expected_action") != "answer" and row.get("action") == "answer"
    if category == "false_reject":
        return row.get("expected_action") == "answer" and row.get("action") != "answer"
    if category == "version_conflict":
        return row.get("reason_code") in {"VERSION_CONFLICT", "VERSION_AMBIGUOUS"}
    if category == "generation_failure":
        return row.get("status") == "failed"
    raise ValueError(f"unknown sample category: {category}")


def _review_item(method: str, row: Mapping[str, object], category: str) -> dict[str, Any]:
    fields = (
        "question_id",
        "question",
        "status",
        "answerable",
        "category",
        "answer_key_points",
        "expected_reason_code",
        "resolved_chunk_ids",
        "requested_version",
        "expected_action",
        "action",
        "reason_code",
        "answer",
        "citations",
        "evidence",
        "error_type",
        "error_message",
    )
    failure_codes = list(_failure_reason_codes(row))
    return {
        "method": method,
        "sample_category": category,
        **{field: row.get(field) for field in fields},
        "generation_status": row.get("status"),
        "judge_status": "not_run",
        "primary_failure_reason": failure_codes[0] if failure_codes else None,
        "failure_reason_codes": failure_codes,
        "failure_reason_confirmed_by": None,
        "review_status": "pending",
        "faithfulness": None,
        "answer_relevance": None,
        "human_notes": None,
        "reviewer": None,
    }


def build_human_review_packet(
    method: str,
    rows: Sequence[Mapping[str, object]],
    *,
    per_category: int = 2,
) -> dict[str, object]:
    if per_category <= 0:
        raise ValueError("per_category must be positive")
    ordered = sorted(rows, key=lambda row: str(row["question_id"]))
    samples = [
        _review_item(method, row, category)
        for category in _SAMPLE_CATEGORIES
        for row in [item for item in ordered if _matches_category(item, category)][
            :per_category
        ]
    ]
    return {
        "schema_version": 1,
        "method": method,
        "human_validation_status": "pending",
        "review_constraints": {
            "evaluable_requires_visible_answer": True,
            "scores": [0, 1, 2],
            "non_evaluable_scores_must_be_null": True,
            "model_generated_reviews_forbidden": True,
            "failure_reason_codes_are_multilabel": True,
            "allowed_failure_reason_codes": [
                "retrieval_miss",
                "dataset_question_ambiguity",
                "generation_contract_failure",
                "upstream_failure",
                "unknown",
            ],
            "dataset_question_ambiguity_requires_human_confirmation": True,
            "judge_failure_does_not_change_generation_review_status": True,
        },
        "samples": samples,
    }


def _review_samples(packet: Mapping[str, object]) -> list[Mapping[str, object]]:
    samples = packet.get("samples")
    if not isinstance(samples, list) or not all(
        isinstance(sample, Mapping) for sample in samples
    ):
        raise TypeError("human review samples must be an array of objects")
    return samples


def resolve_human_review_packet(
    existing: Mapping[str, object] | None,
    template: dict[str, object],
) -> dict[str, object]:
    """Preserve a valid completed review; otherwise return the frozen template."""
    if existing is None or existing.get("human_validation_status") != "complete":
        return template
    if existing.get("schema_version") != template.get("schema_version"):
        raise ValueError("completed human review schema does not match the template")
    if existing.get("method") != template.get("method"):
        raise ValueError("completed human review method does not match the template")
    if existing.get("source_inputs") != template.get("source_inputs"):
        raise ValueError("completed human review sources do not match the template")

    immutable_fields = (
        "question_id",
        "sample_category",
        "status",
        "generation_status",
        "answerable",
        "expected_action",
        "action",
        "answer",
        "error_type",
        "error_message",
    )
    expected_samples = _review_samples(template)
    completed_samples = _review_samples(existing)
    if len(completed_samples) != len(expected_samples):
        raise ValueError("completed human review sample count does not match the template")
    for completed, expected in zip(completed_samples, expected_samples, strict=True):
        if any(completed.get(field) != expected.get(field) for field in immutable_fields):
            raise ValueError("completed human review changed frozen sample fields")

    allowed_codes = set(template["review_constraints"]["allowed_failure_reason_codes"])
    reviewers: set[str] = set()
    for sample in completed_samples:
        reviewer = sample.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("completed human review requires a reviewer for every sample")
        reviewers.add(reviewer)
        status = sample.get("review_status")
        codes = sample.get("failure_reason_codes")
        if not isinstance(codes, list) or not all(isinstance(code, str) for code in codes):
            raise TypeError("failure_reason_codes must be a string array")
        if not set(codes).issubset(allowed_codes):
            raise ValueError("completed human review contains an unknown failure reason")
        if status == "evaluable":
            answer = sample.get("answer")
            scores = (sample.get("faithfulness"), sample.get("answer_relevance"))
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("evaluable review requires a visible answer")
            if any(type(score) is not int or score not in {0, 1, 2} for score in scores):
                raise ValueError("evaluable scores must be integers in {0, 1, 2}")
            if codes or sample.get("primary_failure_reason") is not None:
                raise ValueError("evaluable review cannot contain failure reasons")
        elif status == "non_evaluable":
            if sample.get("faithfulness") is not None or sample.get("answer_relevance") is not None:
                raise ValueError("non-evaluable review scores must be null")
            if not codes or sample.get("primary_failure_reason") not in codes:
                raise ValueError("non-evaluable review requires a primary failure reason")
            confirmer = sample.get("failure_reason_confirmed_by")
            if not isinstance(confirmer, str) or not confirmer.strip():
                raise ValueError("non-evaluable failure reason requires human confirmation")
        else:
            raise ValueError("completed review_status must be evaluable or non_evaluable")
    if len(reviewers) != 1:
        raise ValueError("completed human review must use one consistent reviewer")
    return existing if isinstance(existing, dict) else dict(existing)


def human_review_summary(packet: Mapping[str, object]) -> dict[str, object]:
    status = packet.get("human_validation_status")
    samples = _review_samples(packet)
    if status != "complete":
        return {
            "status": "pending",
            "reviewer": None,
            "reviewed_count": 0,
            "n_evaluable": 0,
            "n_non_evaluable": 0,
            "faithfulness_mean": None,
            "answer_relevance_mean": None,
            "failure_reason_counts": {},
            "failure_reason_counts_are_multilabel": True,
        }
    evaluable = [sample for sample in samples if sample.get("review_status") == "evaluable"]
    non_evaluable = [
        sample for sample in samples if sample.get("review_status") == "non_evaluable"
    ]
    reviewers = {str(sample["reviewer"]) for sample in samples}
    failure_counts = Counter(
        str(code)
        for sample in non_evaluable
        for code in sample["failure_reason_codes"]
    )
    return {
        "status": "complete",
        "reviewer": next(iter(reviewers)),
        "reviewed_count": len(samples),
        "n_evaluable": len(evaluable),
        "n_non_evaluable": len(non_evaluable),
        "faithfulness_mean": sum(int(sample["faithfulness"]) for sample in evaluable)
        / len(evaluable),
        "answer_relevance_mean": sum(
            int(sample["answer_relevance"]) for sample in evaluable
        )
        / len(evaluable),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "failure_reason_counts_are_multilabel": True,
    }


_METHOD_ORDER = (
    "b0",
    "b1",
    "m1",
    "m2",
    "b2",
    "m3",
    "m4",
    "m4_no_version_filter",
    "m4_no_citation_validation",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_method(summary: Mapping[str, object]) -> dict[str, object]:
    retrieval = summary["retrieval"]
    generation = summary["generation"]
    refusal = summary["refusal"]
    latency = summary["latency_ms"]
    assert isinstance(retrieval, Mapping)
    assert isinstance(generation, Mapping)
    assert isinstance(refusal, Mapping)
    assert isinstance(latency, Mapping)
    end_to_end = latency.get("end_to_end", {})
    assert isinstance(end_to_end, Mapping)
    return {
        "method": summary["method"],
        "fixed_total": summary["fixed_total"],
        "success_count": summary["success_count"],
        "failed_count": summary["failed_count"],
        "n_evaluable": summary["n_evaluable"],
        "n_non_evaluable": summary["n_non_evaluable"],
        "n_not_applicable": summary["n_not_applicable"],
        "failure_rate": summary["generation_failure_rate"],
        "recall_at_10": retrieval.get("recall_at_10"),
        "mrr_at_10": retrieval.get("mrr_at_10"),
        "ndcg_at_10": retrieval.get("ndcg_at_10"),
        "citation_precision": generation.get("citation_precision"),
        "citation_recall": generation.get("citation_recall"),
        "refusal_precision": refusal.get("refusal_precision"),
        "refusal_recall": refusal.get("refusal_recall"),
        "refusal_f1": refusal.get("refusal_f1"),
        "false_accept_count": refusal.get("false_accept_count"),
        "false_reject_count": refusal.get("false_reject_count"),
        "latency_p50_ms": end_to_end.get("p50"),
        "latency_p95_ms": end_to_end.get("p95"),
        "api_cost_usd": summary["api_cost_usd"],
    }


def generate_final_report(repo_root: Path) -> dict[str, Any]:
    """Generate deterministic tables and a pending human-review packet only."""
    report_root = repo_root / "reports" / "s12_final"
    raw_root = report_root / "raw"
    protocol_path = repo_root / "configs" / "evaluation" / "s12-final-v1.json"
    protocol = _load_json(protocol_path)
    dataset_path = repo_root / str(protocol["dataset"]["test_path"])
    dataset_rows = _load_jsonl(dataset_path)
    dataset_by_id = {str(row["question_id"]): row for row in dataset_rows}
    bootstrap = protocol["bootstrap"]
    assert isinstance(bootstrap, Mapping)

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    frozen_summaries: dict[str, dict[str, Any]] = {}
    source_files: dict[str, str] = {
        protocol_path.relative_to(repo_root).as_posix(): file_sha256(protocol_path),
        dataset_path.relative_to(repo_root).as_posix(): file_sha256(dataset_path),
    }
    for method in _METHOD_ORDER:
        results_path = raw_root / method / "results.jsonl"
        summary_path = raw_root / method / "summary.json"
        rows_by_method[method] = _load_jsonl(results_path)
        frozen_summaries[method] = _load_json(summary_path)
        source_files[results_path.relative_to(repo_root).as_posix()] = file_sha256(results_path)
        source_files[summary_path.relative_to(repo_root).as_posix()] = file_sha256(summary_path)

    method_summaries = {
        method: build_method_summary(method, rows_by_method[method])
        for method in _METHOD_ORDER
    }
    comparisons: list[dict[str, object]] = []
    for baseline, candidate, label in (
        ("b0", "m1", "M1 minus B0 (add Dense)"),
        ("b1", "m1", "M1 minus B1 (add BM25)"),
        ("m1", "m2", "M2 minus M1 (add Cross-Encoder)"),
    ):
        for metric_name in ("recall_at_10", "mrr_at_10", "ndcg_at_10"):
            comparisons.append(
                {
                    "comparison": label,
                    "baseline": baseline,
                    "candidate": candidate,
                    **build_paired_comparison(
                        rows_by_method[baseline],
                        rows_by_method[candidate],
                        metric_name=metric_name,
                        samples=int(bootstrap["samples"]),
                        seed=int(bootstrap["seed"]),
                        confidence_level=float(bootstrap["confidence_level"]),
                    ),
                }
            )
    for candidate, label in (
        ("m4_no_version_filter", "M4 minus no version filter"),
        ("m4_no_citation_validation", "M4 minus no citation validation"),
    ):
        for metric_name in ("citation_precision", "citation_recall"):
            comparisons.append(
                {
                    "comparison": label,
                    "baseline": candidate,
                    "candidate": "m4",
                    **build_paired_comparison(
                        rows_by_method[candidate],
                        rows_by_method["m4"],
                        metric_name=metric_name,
                        samples=int(bootstrap["samples"]),
                        seed=int(bootstrap["seed"]),
                        confidence_level=float(bootstrap["confidence_level"]),
                    ),
                }
            )

    review_rows = [
        {**dataset_by_id[str(row["question_id"])], **row}
        for row in rows_by_method["m4"]
    ]
    packet_template = build_human_review_packet("m4", review_rows, per_category=2)
    packet_template["source_inputs"] = {
        "results": {
            "path": "reports/s12_final/raw/m4/results.jsonl",
            "sha256": source_files["reports/s12_final/raw/m4/results.jsonl"],
        },
        "dataset": {
            "path": dataset_path.relative_to(repo_root).as_posix(),
            "sha256": source_files[dataset_path.relative_to(repo_root).as_posix()],
        },
    }
    packet_path = report_root / "human_review_packet.json"
    existing_packet = _load_json(packet_path) if packet_path.exists() else None
    packet = resolve_human_review_packet(existing_packet, packet_template)
    if packet is packet_template:
        _write_json(packet_path, packet)
    review_summary = human_review_summary(packet)
    source_files[packet_path.relative_to(repo_root).as_posix()] = file_sha256(packet_path)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        "dataset": {
            "split": "test",
            "fixed_total": 60,
            "sha256": protocol["dataset"]["test_sha256"],
        },
        "human_review_status": review_summary["status"],
        "human_review": review_summary,
        "judge_status": "not_run",
        "review_denominator_contract": {
            "n_evaluable": "successful rows with a visible answer",
            "n_non_evaluable": "generation failures; human answer-quality scores must remain null",
            "n_not_applicable": "retrieval-only rows or successful pre-generation refuse/clarify decisions",
            "fixed_total_identity": "n_evaluable + n_non_evaluable + n_not_applicable = 60",
        },
        "methods": method_summaries,
        "frozen_run_summaries": frozen_summaries,
        "paired_bootstrap": comparisons,
        "source_files": dict(sorted(source_files.items())),
        "interpretation_boundary": (
            "Unseen Test estimation only; no post-Test tuning. Human answer-quality "
            "scores cover only the frozen review packet; Judge-human agreement is not "
            "available because the Test Judge was not run."
        ),
    }
    summary_path = report_root / "summary.json"
    _write_json(summary_path, summary)

    table_root = report_root / "tables"
    method_rows = [_flatten_method(method_summaries[method]) for method in _METHOD_ORDER]
    _write_csv(table_root / "method_comparison.csv", method_rows)
    _write_csv(table_root / "bootstrap_ci.csv", comparisons)
    failure_rows = [
        {
            "method": method,
            "fixed_total": 60,
            "failed_count": method_summaries[method]["failed_count"],
            "generation_failure_rate": method_summaries[method]["generation_failure_rate"],
            "failure_reason": reason,
            "failure_reason_count": count,
            "counts_are_multilabel": True,
            "judge_status": "not_run",
            "judge_failed_count": 0,
        }
        for method in _METHOD_ORDER
        for reason, count in (
            method_summaries[method]["failure_reason_counts"].items()
            or [("none", 0)]
        )
    ]
    _write_csv(table_root / "failure_distribution.csv", failure_rows)

    return summary


def verify_final_report(repo_root: Path) -> dict[str, int]:
    """Verify the frozen S12 delivery without regenerating or mutating it."""
    report_root = repo_root / "reports" / "s12_final"
    raw_root = report_root / "raw"
    ledger = _load_json(report_root / "execution-ledger.json")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or len(entries) != len(_METHOD_ORDER):
        raise ValueError("S12 execution ledger must contain all nine methods")
    entries_by_method = {
        str(entry["method"]): entry
        for entry in entries
        if isinstance(entry, Mapping) and "method" in entry
    }
    if tuple(entry["method"] for entry in entries) != _METHOD_ORDER:
        raise ValueError("S12 execution ledger method order does not match protocol")

    result_row_count = 0
    for method in _METHOD_ORDER:
        results_path = raw_root / method / "results.jsonl"
        run_summary_path = raw_root / method / "summary.json"
        rows = _load_jsonl(results_path)
        run_summary = _load_json(run_summary_path)
        ledger_entry = entries_by_method[method]
        if len(rows) != 60 or len({str(row.get("question_id")) for row in rows}) != 60:
            raise ValueError(f"{method} must contain 60 unique Test records")
        results_hash = file_sha256(results_path)
        summary_hash = file_sha256(run_summary_path)
        if results_hash != run_summary.get("results_sha256"):
            raise ValueError(f"{method} results hash does not match its run summary")
        if results_hash != ledger_entry.get("results_sha256"):
            raise ValueError(f"{method} results hash does not match the execution ledger")
        if summary_hash != ledger_entry.get("summary_sha256"):
            raise ValueError(f"{method} summary hash does not match the execution ledger")
        if run_summary.get("question_count") != 60 or run_summary.get("split") != "test":
            raise ValueError(f"{method} run summary violates the frozen Test contract")
        result_row_count += len(rows)

    summary = _load_json(report_root / "summary.json")
    if summary.get("human_review_status") != "complete":
        raise ValueError("S12 human review must be complete")
    if summary.get("judge_status") != "not_run":
        raise ValueError("S12 Test Judge must remain not_run")
    methods = summary.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(_METHOD_ORDER):
        raise ValueError("S12 summary must contain all frozen methods")
    source_files = summary.get("source_files")
    if not isinstance(source_files, Mapping):
        raise TypeError("S12 summary source_files must be an object")
    for relative, expected_hash in source_files.items():
        source_path = repo_root / str(relative)
        if file_sha256(source_path) != expected_hash:
            raise ValueError(f"S12 source hash mismatch: {relative}")

    protocol = _load_json(repo_root / "configs" / "evaluation" / "s12-final-v1.json")
    dataset_path = repo_root / str(protocol["dataset"]["test_path"])
    dataset_rows = _load_jsonl(dataset_path)
    dataset_by_id = {str(row["question_id"]): row for row in dataset_rows}
    m4_rows = _load_jsonl(raw_root / "m4" / "results.jsonl")
    review_rows = [
        {**dataset_by_id[str(row["question_id"])], **row}
        for row in m4_rows
    ]
    packet_template = build_human_review_packet("m4", review_rows, per_category=2)
    packet_template["source_inputs"] = {
        "results": {
            "path": "reports/s12_final/raw/m4/results.jsonl",
            "sha256": file_sha256(raw_root / "m4" / "results.jsonl"),
        },
        "dataset": {
            "path": dataset_path.relative_to(repo_root).as_posix(),
            "sha256": file_sha256(dataset_path),
        },
    }
    packet = _load_json(report_root / "human_review_packet.json")
    resolve_human_review_packet(packet, packet_template)
    review = human_review_summary(packet)
    if review != summary.get("human_review"):
        raise ValueError("human review summary does not match the completed packet")

    figures_root = report_root / "figures"
    figure_manifest = _load_json(figures_root / "manifest.json")
    figures = figure_manifest.get("figures")
    if not isinstance(figures, Mapping) or len(figures) != 4:
        raise ValueError("S12 figure manifest must contain four figures")
    for figure_name, figure in figures.items():
        if not isinstance(figure, Mapping):
            raise TypeError(f"invalid figure manifest entry: {figure_name}")
        for variant in ("png", "svg"):
            artifact = figure.get(variant)
            if not isinstance(artifact, Mapping):
                raise TypeError(f"missing {variant} artifact for {figure_name}")
            artifact_path = figures_root / str(artifact["path"])
            if file_sha256(artifact_path) != artifact.get("sha256"):
                raise ValueError(f"figure hash mismatch: {artifact_path}")
        png = figure["png"]
        if not isinstance(png, Mapping) or png.get("dpi") != 450:
            raise ValueError(f"figure PNG must be 450 DPI: {figure_name}")
        sources = figure.get("sources")
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError(f"figure sources are missing: {figure_name}")
        for relative, expected_hash in sources.items():
            normalized = str(relative).replace("\\", "/")
            if file_sha256(repo_root / normalized) != expected_hash:
                raise ValueError(f"figure source hash mismatch: {figure_name}")

    with (report_root / "tables" / "method_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        if len(list(csv.DictReader(handle))) != len(_METHOD_ORDER):
            raise ValueError("method comparison table must contain nine rows")

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    results_doc = (repo_root / "docs" / "S12_RESULTS.md").read_text(encoding="utf-8")
    for method in _METHOD_ORDER:
        results_key = f"reports/s12_final/raw/{method}/results.jsonl"
        digest = str(source_files[results_key])
        if digest not in results_doc:
            raise ValueError(f"S12 results document is missing the {method} source hash")
    required_readme_values = (
        "nDCG@10 0.6831",
        "nDCG@10 0.5951",
        "nDCG@10 0.6955",
        "nDCG@10 0.8298",
        "拒答 F1 0.8657",
        "Citation P/R 0.7398/0.8941",
        "58/2",
        "12,848.34 ms",
    )
    if any(value not in readme for value in required_readme_values):
        raise ValueError("README S12 result table is incomplete or inconsistent")
    if "[待实验填写]" in readme or "[待实验填写]" in results_doc:
        raise ValueError("final S12 documents must not contain result placeholders")
    for relative in ("docs/S12_DEMO_SCRIPT.md",):
        if not (repo_root / relative).is_file():
            raise ValueError(f"missing S12 delivery document: {relative}")

    return {
        "method_count": len(methods),
        "result_row_count": result_row_count,
        "reviewed_count": int(review["reviewed_count"]),
        "figure_count": len(figures),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the frozen S12 final report")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify frozen artifacts without regenerating files",
    )
    args = parser.parse_args()
    if args.verify_only:
        verified = verify_final_report(args.repo_root.resolve())
        print(
            "S12_REPORT_VERIFY_PASS "
            f"methods={verified['method_count']} rows={verified['result_row_count']} "
            f"reviews={verified['reviewed_count']} figures={verified['figure_count']}"
        )
        return 0
    summary = generate_final_report(args.repo_root.resolve())
    print(
        "S12_REPORT_PASS "
        f"methods={len(summary['methods'])} bootstrap={len(summary['paired_bootstrap'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
