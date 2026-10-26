"""Immutable one-shot M3/M4 evaluation for the frozen S12 Test split."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from docguard.evaluation.final_protocol import S12TestAuthorization
from docguard.evaluation.generation_metrics import (
    aggregate_generation_metrics,
    citation_metrics,
)
from docguard.evaluation.metrics import summarize_operations
from docguard.evaluation.models import EvaluationQuestion
from docguard.evaluation.refusal_metrics import (
    MetricRecord,
    aggregate_refusal_metrics,
    expected_action,
)
from docguard.evaluation.runner import (
    RunContractError,
    _atomic_text,
    _json_line,
    _read_jsonl_by_id,
    _write_result_views,
)
from docguard.fs_safety import ensure_directory
from docguard.refusal.models import Action, ReasonCode
from docguard.refusal.pipeline import M4Result

_METHODS = {"m3", "m4", "m4_no_version_filter", "m4_no_citation_validation"}
_SECRET_RE = re.compile(r"(?i)(?:secret|api[_-]?key|bearer)(?:\s*[:=]\s*|\s+)[^\s,;]+")


class FinalPipeline(Protocol):
    async def answer(
        self,
        question: str,
        requested_version: str | None,
        **switches: bool,
    ) -> M4Result: ...


def _refusal_score(action: Action, confidence: float) -> float:
    if action is Action.ANSWER:
        return 1.0 - confidence
    if action in {Action.REFUSE, Action.CLARIFY}:
        return confidence
    return 1.0


def _usage(result: M4Result) -> dict[str, object]:
    generated = result.generated
    if generated is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "api_cost_usd": 0.0,
            "api_cost_usd_min": 0.0,
            "api_cost_usd_max": 0.0,
        }
    usage = generated.usage
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "api_cost_usd": usage.api_cost_usd,
        "api_cost_usd_min": usage.api_cost_usd
        if usage.api_cost_usd_min is None
        else usage.api_cost_usd_min,
        "api_cost_usd_max": usage.api_cost_usd
        if usage.api_cost_usd_max is None
        else usage.api_cost_usd_max,
    }


def _latency(result: M4Result, elapsed_ms: float) -> dict[str, float]:
    latency = dict(result.timing_ms)
    generated = result.generated
    if generated is None:
        return {**latency, "end_to_end": elapsed_ms}
    timing = generated.timing_ms
    latency.update(
        {
            "evidence": timing.evidence,
            "generation": timing.generation,
            "validation": timing.validation,
            "end_to_end": elapsed_ms,
        }
    )
    latency.setdefault("retrieval", timing.retrieval)
    return latency


def _result_row(
    question: EvaluationQuestion,
    result: M4Result,
    parameters: Mapping[str, object],
    elapsed_ms: float,
) -> dict[str, object]:
    decision = result.decision
    generated = result.generated
    failed = (
        decision.action is Action.ERROR
        or decision.reason_code is ReasonCode.CITATION_VALIDATION_FAILED
    )
    if decision.action is Action.ANSWER and generated is None:
        failed = True
    citations = (
        [] if generated is None else [item.to_dict() for item in generated.citations]
    )
    evidence = (
        [] if generated is None else [item.to_dict() for item in generated.evidence]
    )
    cited_ids = (
        ()
        if generated is None
        else tuple(item.chunk_id for item in generated.citations)
    )
    known_ids = tuple(
        dict.fromkeys(
            [
                *(
                    item.chunk_id
                    for item in (() if generated is None else generated.evidence)
                ),
                *question.resolved_chunk_ids,
            ]
        )
    )
    metrics = (
        citation_metrics(cited_ids, question.resolved_chunk_ids, known_ids)
        if question.answerable and generated is not None
        else {}
    )
    raw_content = None if generated is None else generated.raw_content
    raw_bytes = None if raw_content is None else raw_content.encode("utf-8")
    return {
        "schema_version": 1,
        "question_id": question.question_id,
        "question": question.question,
        "requested_version": question.requested_version,
        "split": question.split,
        "answerable": question.answerable,
        "route": result.route,
        "status": "failed" if failed else "success",
        "parameters": dict(parameters),
        "action": decision.action.value,
        "expected_action": expected_action(
            question.answerable, question.expected_reason_code
        ).value,
        "reason_code": None
        if decision.reason_code is None
        else decision.reason_code.value,
        "confidence": decision.confidence,
        "triggered_rule": decision.triggered_rule,
        "features": asdict(decision.features),
        "answer": None if generated is None else generated.answer,
        "citations": citations,
        "evidence": evidence,
        "hits": [asdict(item) for item in result.hits],
        "validation_status": None if generated is None else generated.validation_status,
        "raw_content": raw_content,
        "raw_content_sha256": None
        if raw_bytes is None
        else hashlib.sha256(raw_bytes).hexdigest(),
        "raw_content_chars": None if raw_content is None else len(raw_content),
        "raw_content_bytes": None if raw_bytes is None else len(raw_bytes),
        "request_id": None if generated is None else generated.request_id,
        "model": None if generated is None else generated.model,
        "provider_fingerprint": None
        if generated is None
        else generated.provider_fingerprint,
        "metrics": metrics,
        "latency_ms": _latency(result, elapsed_ms),
        "usage": _usage(result),
        "error_type": None
        if not failed
        else (
            "CitationValidationFailure"
            if decision.reason_code is ReasonCode.CITATION_VALIDATION_FAILED
            else "FinalPipelineError"
        ),
        "error_message": None
        if not failed or result.error_message is None
        else _SECRET_RE.sub("[REDACTED]", result.error_message),
    }


def _refusal_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    records = []
    for row in rows:
        confidence = float(row["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("decision confidence must be finite and in [0, 1]")
        predicted = Action(str(row["action"]))
        records.append(
            MetricRecord(
                str(row["question_id"]),
                Action(str(row["expected_action"])),
                predicted,
                _refusal_score(predicted, confidence),
            )
        )
    return asdict(aggregate_refusal_metrics(records))


async def run_final_evaluation(
    questions: Sequence[EvaluationQuestion],
    output_dir: Path,
    pipeline: FinalPipeline,
    metadata: Mapping[str, object],
    *,
    test_authorization: S12TestAuthorization,
) -> dict[str, object]:
    """Run one pre-registered M3/M4 route, never repeating recorded IDs."""
    method = metadata.get("method")
    if method not in _METHODS:
        raise RunContractError("final method is not an allowed M3/M4 S12 route")
    try:
        test_authorization.validate(output_dir, metadata, questions)
    except ValueError as error:
        raise RunContractError(f"S12 authorization failed: {error}") from error
    parameters = metadata.get("parameters")
    if not isinstance(parameters, Mapping):
        raise RunContractError("final method parameters must be an object")
    metadata_dict = dict(metadata)
    ensure_directory(output_dir)
    run_path = output_dir / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text(encoding="utf-8")) != metadata_dict:
            raise RunContractError(
                "run metadata differs from the existing run metadata"
            )
    else:
        _atomic_text(run_path, _json_line(metadata_dict))
    rows = _read_jsonl_by_id(output_dir / "results.jsonl")
    expected_ids = {question.question_id for question in questions}
    if set(rows) - expected_ids:
        raise RunContractError("saved results contain unexpected question IDs")
    switches = {
        name: value
        for name, value in parameters.items()
        if name
        in {
            "refusal_enabled",
            "version_filter_enabled",
            "generation_enabled",
            "citation_validation_enabled",
        }
    }
    if len(switches) != 4 or any(
        type(value) is not bool for value in switches.values()
    ):
        raise RunContractError("final method switches are incomplete")
    for question in sorted(questions, key=lambda item: item.question_id):
        if question.question_id in rows:
            continue
        started = time.perf_counter()
        try:
            result = await pipeline.answer(
                question.question, question.requested_version, **switches
            )
            rows[question.question_id] = _result_row(
                question,
                result,
                parameters,
                (time.perf_counter() - started) * 1000.0,
            )
        except Exception as error:  # noqa: BLE001 - one failed item must not hide others
            elapsed = (time.perf_counter() - started) * 1000.0
            rows[question.question_id] = {
                "schema_version": 1,
                "question_id": question.question_id,
                "question": question.question,
                "requested_version": question.requested_version,
                "split": "test",
                "answerable": question.answerable,
                "route": method,
                "status": "failed",
                "parameters": dict(parameters),
                "action": Action.ERROR.value,
                "expected_action": expected_action(
                    question.answerable, question.expected_reason_code
                ).value,
                "reason_code": ReasonCode.UPSTREAM_ERROR.value,
                "confidence": 1.0,
                "triggered_rule": "runner_exception",
                "features": {},
                "answer": None,
                "citations": [],
                "evidence": [],
                "hits": [],
                "validation_status": None,
                "raw_content": None,
                "raw_content_sha256": None,
                "raw_content_chars": None,
                "raw_content_bytes": None,
                "request_id": None,
                "model": None,
                "provider_fingerprint": None,
                "metrics": {},
                "latency_ms": {"end_to_end": elapsed},
                "usage": {
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "api_cost_usd": 0.0,
                    "api_cost_usd_min": 0.0,
                    "api_cost_usd_max": 0.0,
                },
                "error_type": type(error).__name__,
                "error_message": _SECRET_RE.sub("[REDACTED]", str(error)),
            }
        _write_result_views(output_dir, rows)
    ordered = [rows[key] for key in sorted(rows)]
    generated_rows = [row for row in ordered if row.get("answer") is not None]
    summary = {
        "schema_version": 1,
        "protocol_version": metadata_dict.get("protocol_version"),
        "run_id": metadata_dict.get("run_id"),
        "method": method,
        "split": "test",
        "dataset_version": metadata_dict.get("dataset_version"),
        "dataset_sha256": metadata_dict.get("dataset_sha256"),
        "results_sha256": hashlib.sha256(
            (output_dir / "results.jsonl").read_bytes()
        ).hexdigest(),
        "question_count": len(questions),
        "refusal": _refusal_summary(ordered),
        "generation": aggregate_generation_metrics(generated_rows),
        "operations": summarize_operations(ordered),
    }
    _atomic_text(output_dir / "summary.json", _json_line(summary))
    return summary
