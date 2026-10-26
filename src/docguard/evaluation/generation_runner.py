"""Immutable, resumable S7 generation evaluation with a hard Test lock."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from docguard.evaluation.final_protocol import S12TestAuthorization
from docguard.evaluation.generation_metrics import (
    aggregate_generation_metrics,
    citation_metrics,
)
from docguard.evaluation.metrics import summarize_operations
from docguard.evaluation.models import EvaluationQuestion
from docguard.evaluation.runner import (
    RunContractError,
    TestSplitLockedError,
    _atomic_text,
    _json_line,
    _read_jsonl_by_id,
    _write_result_views,
)
from docguard.fs_safety import ensure_directory
from docguard.generation.config import GenerationConfig
from docguard.generation.models import GeneratedAnswer
from docguard.generation.service import PostResponseGenerationError

_SECRET_RE = re.compile(r"(?i)(?:secret|api[_-]?key|bearer)(?:\s*[:=]\s*|\s+)[^\s,;]+")
_SECRET_FIELD_RE = re.compile(
    r"(?i)(?:^|[_-])(?:secret|api[_-]?key|bearer)(?:$|[_-])"
)


class GenerationPipeline(Protocol):
    async def answer(
        self,
        question: str,
        requested_version: str | None,
        *,
        generation_enabled: bool,
        citation_validation_enabled: bool,
    ) -> GeneratedAnswer: ...


def _redact_json_value(value: object) -> tuple[object, bool]:
    if isinstance(value, dict):
        redacted: dict[object, object] = {}
        changed = False
        for key, item in value.items():
            if isinstance(key, str) and _SECRET_FIELD_RE.search(key):
                redacted[key] = "[REDACTED]"
                changed = True
            else:
                redacted_item, item_changed = _redact_json_value(item)
                redacted[key] = redacted_item
                changed = changed or item_changed
        return redacted, changed
    if isinstance(value, list):
        redacted_items = []
        changed = False
        for item in value:
            redacted_item, item_changed = _redact_json_value(item)
            redacted_items.append(redacted_item)
            changed = changed or item_changed
        return redacted_items, changed
    if isinstance(value, str):
        redacted = _SECRET_RE.sub("[REDACTED]", value)
        return redacted, redacted != value
    return value, False


def _raw_content_audit(
    value: str | None, *, include_content: bool
) -> dict[str, object]:
    if value is None:
        return {
            "raw_content": None,
            "raw_content_sha256": None,
            "raw_content_chars": None,
            "raw_content_bytes": None,
        }
    encoded = value.encode("utf-8")
    if not include_content:
        saved_content = None
    else:
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            saved_content = None
        else:
            redacted, changed = _redact_json_value(payload)
            saved_content = (
                json.dumps(redacted, ensure_ascii=False, separators=(",", ":"))
                if changed
                else value
            )
    return {
        "raw_content": saved_content,
        "raw_content_sha256": hashlib.sha256(encoded).hexdigest(),
        "raw_content_chars": len(value),
        "raw_content_bytes": len(encoded),
    }


def _redact_error(error: Exception) -> str:
    return _SECRET_RE.sub("[REDACTED]", str(error))


def _validate_contract(
    questions: Sequence[EvaluationQuestion],
    config: GenerationConfig,
    metadata: Mapping[str, object],
    output_dir: Path,
    test_authorization: S12TestAuthorization | None,
) -> str:
    split = metadata.get("split")
    if split == "test" or any(question.split == "test" for question in questions):
        if split != "test" or any(question.split != "test" for question in questions):
            raise RunContractError("all Test questions must match the run split")
        if test_authorization is None:
            raise TestSplitLockedError("Test split requires the frozen S12 authorization")
        try:
            test_authorization.validate(output_dir, metadata, questions)
        except ValueError as error:
            raise RunContractError(f"S12 authorization failed: {error}") from error
    elif test_authorization is not None:
        raise RunContractError("S12 Test authorization cannot be used for a non-Test run")
    elif split not in {"development", "validation"}:
        raise RunContractError("split must be development or validation")
    if not questions or any(question.split != split for question in questions):
        raise RunContractError("all questions must match the non-empty run split")
    identities = {
        "dataset_version": config.dataset_version,
        "generation_version": config.generation_version,
        "provider": config.provider,
        "protocol": config.protocol,
        "model": config.model,
        "documented_model_version": config.documented_model_version,
        "model_version_checked_on": config.model_version_checked_on,
        "prompt_version": config.prompt_version,
        "prompt_sha256": config.prompt_sha256,
        "output_schema_sha256": config.output_schema_sha256,
        "base_url": config.default_base_url,
        "minimum_input_cost_per_million_usd": config.minimum_input_cost_per_million_usd,
        "input_cost_per_million_usd": config.input_cost_per_million_usd,
        "minimum_cached_input_cost_per_million_usd": config.minimum_cached_input_cost_per_million_usd,
        "cached_input_cost_per_million_usd": config.cached_input_cost_per_million_usd,
        "minimum_output_cost_per_million_usd": config.minimum_output_cost_per_million_usd,
        "output_cost_per_million_usd": config.output_cost_per_million_usd,
        "pricing_checked_on": config.pricing_checked_on,
    }
    for field, expected in identities.items():
        if metadata.get(field) != expected:
            raise RunContractError(
                f"{field} does not match the frozen generation config"
            )
    for field in ("dataset_sha256", "generation_config_sha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RunContractError(f"{field} must be a lowercase SHA-256")
    if metadata.get("route") != "b2":
        raise RunContractError("route must be b2")
    parameters = metadata.get("parameters")
    expected_parameters = {
        "dense_enabled": True,
        "bm25_enabled": False,
        "rerank_enabled": False,
        "generation_enabled": True,
        "citation_validation_enabled": True,
        "top_k": config.evidence_top_k,
    }
    if parameters != expected_parameters:
        raise RunContractError("B2 route switches do not match the frozen contract")
    return split


def _success_row(
    question: EvaluationQuestion, generated: GeneratedAnswer
) -> dict[str, object]:
    cited_ids = tuple(citation.chunk_id for citation in generated.citations)
    known_ids = tuple(
        dict.fromkeys(
            [
                *(item.chunk_id for item in generated.evidence),
                *question.resolved_chunk_ids,
            ]
        )
    )
    metrics = (
        citation_metrics(cited_ids, question.resolved_chunk_ids, known_ids)
        if question.answerable
        else {}
    )
    timing = generated.timing_ms
    return {
        "schema_version": 1,
        "question_id": question.question_id,
        "question": question.question,
        "requested_version": question.requested_version,
        "split": question.split,
        "answerable": question.answerable,
        "route": "b2",
        "status": "success",
        "answer": generated.answer,
        **_raw_content_audit(generated.raw_content, include_content=True),
        "citations": [item.to_dict() for item in generated.citations],
        "evidence": [item.to_dict() for item in generated.evidence],
        "validation_status": generated.validation_status,
        "request_id": generated.request_id,
        "model": generated.model,
        "provider_fingerprint": generated.provider_fingerprint,
        "metrics": metrics,
        "latency_ms": {
            "retrieval": timing.retrieval,
            "evidence": timing.evidence,
            "generation": timing.generation,
            "validation": timing.validation,
            "end_to_end": timing.end_to_end,
        },
        "usage": {
            "input_tokens": generated.usage.input_tokens,
            "cached_input_tokens": generated.usage.cached_input_tokens,
            "output_tokens": generated.usage.output_tokens,
            "api_cost_usd": generated.usage.api_cost_usd,
            "api_cost_usd_min": (
                generated.usage.api_cost_usd
                if generated.usage.api_cost_usd_min is None
                else generated.usage.api_cost_usd_min
            ),
            "api_cost_usd_max": (
                generated.usage.api_cost_usd
                if generated.usage.api_cost_usd_max is None
                else generated.usage.api_cost_usd_max
            ),
        },
        "error_type": None,
        "error_message": None,
    }


def _failure_row(
    question: EvaluationQuestion, error: Exception, elapsed_ms: float
) -> dict[str, object]:
    if isinstance(error, PostResponseGenerationError):
        context = error.context
        timing = context.timing_ms
        usage = context.usage
        return {
            "schema_version": 1,
            "question_id": question.question_id,
            "question": question.question,
            "requested_version": question.requested_version,
            "split": question.split,
            "answerable": question.answerable,
            "route": "b2",
            "status": "failed",
            "answer": None,
            **_raw_content_audit(context.raw_content, include_content=False),
            "citations": [],
            "evidence": [item.to_dict() for item in context.evidence],
            "validation_status": None,
            "request_id": context.request_id,
            "model": context.model,
            "provider_fingerprint": context.provider_fingerprint,
            "metrics": {},
            "latency_ms": {
                "retrieval": timing.retrieval,
                "evidence": timing.evidence,
                "generation": timing.generation,
                "validation": timing.validation,
                "end_to_end": timing.end_to_end,
            },
            "usage": {
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "output_tokens": usage.output_tokens,
                "api_cost_usd": usage.api_cost_usd,
                "api_cost_usd_min": (
                    usage.api_cost_usd
                    if usage.api_cost_usd_min is None
                    else usage.api_cost_usd_min
                ),
                "api_cost_usd_max": (
                    usage.api_cost_usd
                    if usage.api_cost_usd_max is None
                    else usage.api_cost_usd_max
                ),
            },
            "error_type": type(error.validation_error).__name__,
            "error_message": _redact_error(error.validation_error),
        }
    return {
        "schema_version": 1,
        "question_id": question.question_id,
        "question": question.question,
        "requested_version": question.requested_version,
        "split": question.split,
        "answerable": question.answerable,
        "route": "b2",
        "status": "failed",
        "answer": None,
        **_raw_content_audit(None, include_content=False),
        "citations": [],
        "evidence": [],
        "validation_status": None,
        "request_id": None,
        "model": None,
        "provider_fingerprint": None,
        "metrics": {},
        "latency_ms": {"end_to_end": elapsed_ms},
        "usage": {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "api_cost_usd": 0.0,
            "api_cost_usd_min": 0.0,
            "api_cost_usd_max": 0.0,
        },
        "error_type": type(error).__name__,
        "error_message": _redact_error(error),
    }


def _operations(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    operations = summarize_operations(rows)
    cached = 0
    minimum_cost = 0.0
    maximum_cost = 0.0
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            raise TypeError("rows require usage objects")
        value = usage.get("cached_input_tokens")
        if type(value) is not int or value < 0:
            raise ValueError("cached_input_tokens must be a non-negative integer")
        cached += value
        for field in ("api_cost_usd_min", "api_cost_usd_max"):
            cost = usage.get(field)
            if (
                type(cost) not in {int, float}
                or not math.isfinite(float(cost))
                or float(cost) < 0
            ):
                raise ValueError(f"{field} must be finite and non-negative")
        minimum_cost += float(usage["api_cost_usd_min"])
        maximum_cost += float(usage["api_cost_usd_max"])
    operations_usage = operations["usage"]
    if not isinstance(operations_usage, dict):
        raise TypeError("operation usage must be an object")
    operations_usage["cached_input_tokens"] = cached
    operations_usage["api_cost_usd_min"] = minimum_cost
    operations_usage["api_cost_usd_max"] = maximum_cost
    return operations


async def run_generation_evaluation(
    questions: Sequence[EvaluationQuestion],
    output_dir: Path,
    pipeline: GenerationPipeline,
    config: GenerationConfig,
    metadata: Mapping[str, object],
    *,
    test_authorization: S12TestAuthorization | None = None,
) -> dict[str, object]:
    """Run or resume B2, persisting each question atomically."""
    split = _validate_contract(
        questions, config, metadata, output_dir, test_authorization
    )
    metadata_dict = dict(metadata)
    ensure_directory(output_dir)
    run_path = output_dir / "run.json"
    if run_path.exists():
        saved = json.loads(run_path.read_text(encoding="utf-8"))
        if saved != metadata_dict:
            raise RunContractError(
                "run metadata differs from the existing run metadata"
            )
    else:
        _atomic_text(run_path, _json_line(metadata_dict))
    results_path = output_dir / "results.jsonl"
    rows = _read_jsonl_by_id(results_path)
    expected_ids = {question.question_id for question in questions}
    if set(rows) - expected_ids:
        raise RunContractError("saved results contain unexpected question IDs")
    for question in sorted(questions, key=lambda item: item.question_id):
        if question.question_id in rows:
            continue
        started = time.perf_counter()
        try:
            generated = await pipeline.answer(
                question.question,
                question.requested_version,
                generation_enabled=True,
                citation_validation_enabled=True,
            )
            if generated.status != "success":
                raise RunContractError(
                    "generation pipeline returned a non-success result"
                )
            rows[question.question_id] = _success_row(question, generated)
        except Exception as error:  # noqa: BLE001 - failures are isolated per question
            rows[question.question_id] = _failure_row(
                question, error, (time.perf_counter() - started) * 1000.0
            )
        _write_result_views(output_dir, rows)
    ordered = [rows[key] for key in sorted(rows)]
    results_sha256 = hashlib.sha256(results_path.read_bytes()).hexdigest()
    summary: dict[str, object] = {
        "schema_version": 1,
        "run_id": metadata_dict.get("run_id"),
        "split": split,
        "dataset_version": metadata_dict.get("dataset_version"),
        "dataset_sha256": metadata_dict.get("dataset_sha256"),
        "generation_config_sha256": metadata_dict.get("generation_config_sha256"),
        "results_sha256": results_sha256,
        "question_count": len(questions),
        "generation": aggregate_generation_metrics(ordered),
        "operations": _operations(ordered),
    }
    _atomic_text(output_dir / "summary.json", _json_line(summary))
    return summary
