"""Deterministic S8 calibration and immutable result storage."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from docguard.evaluation.refusal_metrics import (
    MetricRecord,
    RefusalMetrics,
    aggregate_refusal_metrics,
)
from docguard.evaluation.runner import TestSplitLockedError, _atomic_text, _json_line
from docguard.fs_safety import ensure_directory
from docguard.refusal.config import RefusalConfig
from docguard.refusal.models import Action, RefusalFeatures
from docguard.refusal.policy import RefusalThresholds, decide
from docguard.refusal.versioning import VersionRequest

__all__ = ["TestSplitLockedError"]


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    question_id: str
    features: RefusalFeatures
    version_request: VersionRequest
    expected_action: Action
    question: str = ""
    category: str = ""
    historical_reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    thresholds: RefusalThresholds
    metrics: RefusalMetrics


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    selected: CalibrationCandidate
    candidates: tuple[CalibrationCandidate, ...]


def _refusal_score(action: Action, confidence: float) -> float:
    if action is Action.ANSWER:
        return 1.0 - confidence
    if action in {Action.REFUSE, Action.CLARIFY}:
        return confidence
    return 1.0


def calibrate_thresholds(
    examples: list[CalibrationExample], config: RefusalConfig
) -> CalibrationSelection:
    if not examples or len({item.question_id for item in examples}) != len(examples):
        raise ValueError("calibration examples must be non-empty with unique IDs")
    candidates: list[CalibrationCandidate] = []
    for values in itertools.product(
        config.threshold_grid.top1,
        config.threshold_grid.top3_mean,
        config.threshold_grid.coverage,
        config.threshold_grid.margin,
    ):
        thresholds = RefusalThresholds(*values)
        records: list[MetricRecord] = []
        for example in sorted(examples, key=lambda item: item.question_id):
            decision = decide(example.features, example.version_request, thresholds)
            records.append(
                MetricRecord(
                    example.question_id,
                    example.expected_action,
                    decision.action,
                    _refusal_score(decision.action, decision.confidence),
                )
            )
        candidates.append(
            CalibrationCandidate(thresholds, aggregate_refusal_metrics(records))
        )

    def selection_key(candidate: CalibrationCandidate) -> tuple[object, ...]:
        metrics = candidate.metrics
        precision = -1.0 if metrics.refusal_precision is None else metrics.refusal_precision
        coverage = -1.0 if metrics.answerable_coverage is None else metrics.answerable_coverage
        return (
            -metrics.macro_f1,
            metrics.false_accept_count,
            -precision,
            -coverage,
            candidate.thresholds,
        )

    ordered = tuple(sorted(candidates, key=lambda item: item.thresholds))
    return CalibrationSelection(min(ordered, key=selection_key), ordered)


def assert_non_test_input(path: Path, split: str) -> None:
    if split.lower() == "test" or path.stem.lower() == "test" or any(
        part.lower() == "test" for part in path.parts
    ):
        raise TestSplitLockedError("Test split is frozen and cannot run before S12")
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
            if isinstance(row, dict) and str(row.get("split", "")).lower() == "test":
                raise TestSplitLockedError("Test content is frozen and cannot run before S12")


class RefusalRunStore:
    """Rewrite stable JSONL views and never rerun any recorded question ID."""

    def __init__(self, output_dir: Path, metadata: dict[str, object]) -> None:
        if metadata.get("split") == "test":
            raise TestSplitLockedError("Test split is frozen and cannot run before S12")
        self.output_dir = output_dir
        ensure_directory(output_dir)
        run_path = output_dir / "run.json"
        stable = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if run_path.exists():
            existing = json.loads(run_path.read_text(encoding="utf-8"))
            if existing != metadata:
                raise ValueError("existing run metadata is immutable")
        else:
            _atomic_text(run_path, stable + "\n")
        self._rows: dict[str, dict[str, object]] = {}
        results_path = output_dir / "results.jsonl"
        if results_path.exists():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                question_id = row.get("question_id")
                if not isinstance(question_id, str) or question_id in self._rows:
                    raise ValueError("saved results contain invalid or duplicate IDs")
                self._rows[question_id] = row

    def append(self, row: dict[str, object]) -> bool:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("result requires a non-empty question_id")
        if question_id in self._rows:
            return False
        self._rows[question_id] = dict(row)
        ordered = [self._rows[key] for key in sorted(self._rows)]
        _atomic_text(
            self.output_dir / "results.jsonl",
            "".join(_json_line(item) for item in ordered),
        )
        failures = [item for item in ordered if item.get("status") == "failed"]
        _atomic_text(
            self.output_dir / "failures.jsonl",
            "".join(_json_line(item) for item in failures),
        )
        return True
