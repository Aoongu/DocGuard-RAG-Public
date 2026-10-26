"""Offline CLI for S8 refusal calibration and saved-result replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from docguard.evaluation.refusal_metrics import (
    MetricRecord,
    aggregate_refusal_metrics,
    expected_action,
)
from docguard.evaluation.refusal_runner import (
    CalibrationExample,
    RefusalRunStore,
    assert_non_test_input,
    calibrate_thresholds,
)
from docguard.evaluation.runner import TestSplitLockedError, _atomic_text, _json_line
from docguard.refusal.config import RefusalConfig, load_refusal_config
from docguard.refusal.features import extract_refusal_features
from docguard.refusal.models import RankedEvidence
from docguard.refusal.policy import RefusalThresholds, decide
from docguard.refusal.versioning import detect_version_conflict, parse_version_request


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _expected_identity(config: RefusalConfig, name: str) -> str:
    identities = dict(config.input_sha256)
    try:
        return identities[name]
    except KeyError as error:
        raise ValueError(f"frozen config is missing input identity: {name}") from error


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 differs from the frozen config")


def load_examples(
    dataset_path: Path,
    results_path: Path,
    chunks_path: Path,
    chunk_manifest_path: Path,
    split: str,
    config: RefusalConfig,
) -> list[CalibrationExample]:
    """Join frozen M2 hit IDs to S2 chunks without rerunning retrieval."""
    assert_non_test_input(dataset_path, split)
    assert_non_test_input(results_path, split)
    if split not in {"development", "validation"}:
        raise ValueError("split must be development or validation")
    prefix = "development" if split == "development" else "validation"
    _verify_hash(dataset_path, _expected_identity(config, f"{prefix}_dataset"), "dataset")
    _verify_hash(
        results_path,
        _expected_identity(config, f"{prefix}_m2_results"),
        "M2 results",
    )
    _verify_hash(
        chunk_manifest_path,
        _expected_identity(config, "s2_chunk_manifest"),
        "chunk manifest",
    )
    manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("chunks_sha256"), str):
        raise TypeError("chunk manifest is missing chunks_sha256")
    _verify_hash(chunks_path, manifest["chunks_sha256"], "chunks")

    questions: dict[str, dict[str, Any]] = {}
    for row in _jsonl(dataset_path):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or question_id in questions:
            raise ValueError("dataset contains invalid or duplicate question IDs")
        if row.get("split") != split:
            raise ValueError("dataset row split differs from the requested split")
        questions[question_id] = row
    results: dict[str, dict[str, Any]] = {}
    required_chunks: set[str] = set()
    for row in _jsonl(results_path):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or question_id in results:
            raise ValueError("M2 results contain invalid or duplicate question IDs")
        if row.get("split") != split or row.get("status") != "success":
            raise ValueError("M2 results must be complete successful rows for the split")
        hits = row.get("hits")
        if not isinstance(hits, list) or not hits:
            raise ValueError("M2 result row has no hits")
        for hit in hits:
            if not isinstance(hit, dict) or not isinstance(hit.get("chunk_id"), str):
                raise TypeError("M2 hit is invalid")
            required_chunks.add(hit["chunk_id"])
        results[question_id] = row
    if set(questions) != set(results):
        raise ValueError("dataset and M2 results contain different question IDs")

    chunks: dict[str, dict[str, Any]] = {}
    for row in _jsonl(chunks_path):
        chunk_id = row.get("chunk_id")
        if chunk_id in required_chunks:
            if not isinstance(chunk_id, str) or chunk_id in chunks:
                raise ValueError("chunks contain an invalid or duplicate required ID")
            chunks[chunk_id] = row
    missing = required_chunks - set(chunks)
    if missing:
        raise ValueError(f"M2 results reference unknown chunks: {sorted(missing)[:3]}")

    examples: list[CalibrationExample] = []
    for question_id in sorted(questions):
        question = questions[question_id]
        ranked: list[RankedEvidence] = []
        for hit in results[question_id]["hits"]:
            chunk = chunks[hit["chunk_id"]]
            ranked.append(
                RankedEvidence(
                    rank=int(hit["rank"]),
                    chunk_id=hit["chunk_id"],
                    rerank_score=float(hit["rerank_score"]),
                    postgresql_version=str(chunk["postgresql_version"]),
                    canonical_url=str(chunk["canonical_url"]),
                    anchor=str(chunk["anchor"]),
                    text=str(chunk["text"]),
                )
            )
        question_text = str(question["question"])
        structured_version = question.get("requested_version")
        if structured_version is not None and not isinstance(structured_version, str):
            raise ValueError("requested_version must be null or a string")
        version_request = parse_version_request(question_text, structured_version, config)
        conflict = detect_version_conflict(ranked, config)
        features = extract_refusal_features(
            question_text,
            ranked,
            config,
            requested_version=version_request.requested_version,
            requested_version_status=version_request.status,
            paired_version_conflict=conflict.detected,
        )
        answerable = question.get("answerable")
        if type(answerable) is not bool:
            raise ValueError("answerable must be boolean")
        reason = question.get("expected_reason_code")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("expected_reason_code must be null or a string")
        examples.append(
            CalibrationExample(
                question_id,
                features,
                version_request,
                expected_action(answerable, reason),
                question_text,
                str(question.get("category", "")),
                reason,
            )
        )
    return examples


def _metrics_dict(value: object) -> dict[str, object]:
    result = asdict(value)
    result["risk_coverage"] = result.pop("risk_coverage")
    return result


def _thresholds_dict(value: object) -> dict[str, object]:
    return asdict(value)


def run_calibration(
    examples: list[CalibrationExample],
    output_dir: Path,
    config: RefusalConfig,
    *,
    run_id: str,
) -> dict[str, object]:
    selection = calibrate_thresholds(examples, config)
    config_payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    metadata: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "split": "validation",
        "refusal_version": config.refusal_version,
        "refusal_config_sha256": hashlib.sha256(config_payload.encode()).hexdigest(),
        "result_status": "selection_result",
        "test_execution_status": config.test_execution_status,
    }
    store = RefusalRunStore(output_dir, metadata)
    thresholds = selection.selected.thresholds
    for example in sorted(examples, key=lambda item: item.question_id):
        decision = decide(example.features, example.version_request, thresholds)
        store.append(
            {
                "schema_version": 1,
                "question_id": example.question_id,
                "split": "validation",
                "status": "success",
                "category": example.category,
                "expected_action": example.expected_action.value,
                "historical_reason_code": example.historical_reason_code,
                "action": decision.action.value,
                "reason_code": None if decision.reason_code is None else decision.reason_code.value,
                "confidence": decision.confidence,
                "triggered_rule": decision.triggered_rule,
                "features": asdict(decision.features),
            }
        )
    selection_payload = {
        "schema_version": 1,
        "selection_split": "validation",
        "result_status": "selection_result",
        "selected": {
            "thresholds": _thresholds_dict(selection.selected.thresholds),
            "metrics": _metrics_dict(selection.selected.metrics),
        },
        "candidates": [
            {
                "thresholds": _thresholds_dict(item.thresholds),
                "macro_f1": item.metrics.macro_f1,
                "false_accept_count": item.metrics.false_accept_count,
                "refusal_precision": item.metrics.refusal_precision,
                "answerable_coverage": item.metrics.answerable_coverage,
            }
            for item in selection.candidates
        ],
    }
    _atomic_text(
        output_dir / "selection.json",
        json.dumps(selection_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    summary: dict[str, object] = {
        **metadata,
        "question_count": len(examples),
        "selected_thresholds": _thresholds_dict(thresholds),
        "metrics": _metrics_dict(selection.selected.metrics),
        "interpretation_limit": (
            "Validation selected these thresholds; this is not an unseen-Test generalization result."
        ),
    }
    _atomic_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    _atomic_text(
        output_dir / "risk_coverage.jsonl",
        "".join(
            _json_line(point)
            for point in selection_payload["selected"]["metrics"]["risk_coverage"]  # type: ignore[index]
        ),
    )
    return summary


def _decision_refusal_score(action: str, confidence: float) -> float:
    return 1.0 - confidence if action == "answer" else confidence


def run_diagnostic(
    examples: list[CalibrationExample],
    output_dir: Path,
    config: RefusalConfig,
    selection_path: Path,
    *,
    run_id: str,
    split: str,
) -> dict[str, object]:
    if split == "test":
        raise TestSplitLockedError("Test split is frozen and cannot run before S12")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    try:
        selected = selection["selected"]
        values = selected["thresholds"]
        thresholds = RefusalThresholds(
            top1=float(values["top1"]),
            top3_mean=float(values["top3_mean"]),
            coverage=float(values["coverage"]),
            margin=float(values["margin"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection file does not contain valid frozen thresholds") from error
    selection_sha256 = _sha256(selection_path)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "split": split,
        "refusal_version": config.refusal_version,
        "selection_sha256": selection_sha256,
        "result_status": "diagnostic_replay",
        "test_execution_status": config.test_execution_status,
    }
    store = RefusalRunStore(output_dir, metadata)
    records: list[MetricRecord] = []
    for example in sorted(examples, key=lambda item: item.question_id):
        decision = decide(example.features, example.version_request, thresholds)
        store.append(
            {
                "schema_version": 1,
                "question_id": example.question_id,
                "split": split,
                "status": "success",
                "category": example.category,
                "expected_action": example.expected_action.value,
                "historical_reason_code": example.historical_reason_code,
                "action": decision.action.value,
                "reason_code": None if decision.reason_code is None else decision.reason_code.value,
                "confidence": decision.confidence,
                "triggered_rule": decision.triggered_rule,
                "features": asdict(decision.features),
            }
        )
        records.append(
            MetricRecord(
                example.question_id,
                example.expected_action,
                decision.action,
                _decision_refusal_score(decision.action.value, decision.confidence),
            )
        )
    metrics = aggregate_refusal_metrics(records)
    summary: dict[str, object] = {
        **metadata,
        "question_count": len(examples),
        "selected_thresholds": asdict(thresholds),
        "metrics": _metrics_dict(metrics),
        "interpretation_limit": (
            "Development is a diagnostic replay of Validation-selected thresholds and does not alter them."
        ),
    }
    _atomic_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    _atomic_text(
        output_dir / "risk_coverage.jsonl",
        "".join(_json_line(asdict(point)) for point in metrics.risk_coverage),
    )
    return summary


def summarize_output(output_dir: Path) -> dict[str, object]:
    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = _jsonl(output_dir / "results.jsonl")
    identifiers = [row.get("question_id") for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("saved output contains duplicate question IDs")
    if summary.get("run_id") != run.get("run_id"):
        raise ValueError("run and summary identities differ")
    if summary.get("question_count") != len(rows):
        raise ValueError("summary question count differs from recorded rows")
    return {
        "run_id": run.get("run_id"),
        "result_status": summary.get("result_status"),
        "question_count": summary.get("question_count"),
        "recorded_count": len(rows),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docguard-refusal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    def add_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--split", required=True)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--results", type=Path, required=True)
        command.add_argument("--chunks", type=Path, required=True)
        command.add_argument("--chunk-manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--config", type=Path, default=Path("configs/refusal/s8-refusal-v1.json")
        )

    calibrate = subparsers.add_parser("calibrate")
    add_inputs(calibrate)
    calibrate.add_argument("--run-id", default="s8-validation-selection-v1")
    diagnostic = subparsers.add_parser("decide")
    add_inputs(diagnostic)
    diagnostic.add_argument("--selection", type=Path, required=True)
    diagnostic.add_argument("--run-id", default="s8-development-diagnostic-v1")
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize":
        summarize_output(args.output)
        return 0
    if args.split.lower() == "test":
        raise TestSplitLockedError("Test split is frozen and cannot run before S12")
    config = load_refusal_config(args.config)
    examples = load_examples(
        args.dataset,
        args.results,
        args.chunks,
        args.chunk_manifest,
        args.split,
        config,
    )
    if args.command == "calibrate":
        if args.split != "validation":
            raise ValueError("calibrate may only select thresholds on Validation")
        run_calibration(examples, args.output, config, run_id=args.run_id)
        return 0
    if args.command == "decide":
        run_diagnostic(
            examples,
            args.output,
            config,
            args.selection,
            run_id=args.run_id,
            split=args.split,
        )
        return 0
    raise ValueError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
