"""Saved-result-only S6 Validation depth selection and paired comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from docguard.evaluation.compare import paired_bootstrap
from docguard.evaluation.config import EvaluationConfig, load_evaluation_config
from docguard.fs_safety import (
    ensure_directory,
    validate_directory,
    validate_regular_file,
)

_METRICS = ("ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_at_5")
_DEPTHS = frozenset({10, 20, 50})


class SelectionError(RuntimeError):
    """Raised when saved S6 runs are incomplete or incomparable."""


@dataclass(frozen=True, slots=True)
class SavedRun:
    root: Path
    run: dict[str, object]
    summary: dict[str, object]
    results: dict[str, dict[str, object]]


def _read_object(path: Path) -> dict[str, object]:
    validate_regular_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionError(f"saved JSON must be an object: {path}")
    return value


def _load(root: Path) -> SavedRun:
    try:
        validate_directory(root)
        run = _read_object(root / "run.json")
        summary = _read_object(root / "summary.json")
        results_path = root / "results.jsonl"
        validate_regular_file(results_path)
        results: dict[str, dict[str, object]] = {}
        for line_number, line in enumerate(
            results_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
                raise SelectionError(f"invalid result row at {root}:{line_number}")
            question_id = row["question_id"]
            if question_id in results:
                raise SelectionError(f"duplicate result question ID: {question_id}")
            if row.get("status") != "success":
                raise SelectionError(f"saved run contains failed row: {root}:{question_id}")
            results[question_id] = row
    except SelectionError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SelectionError(f"unable to load saved run: {root}") from error
    if not results:
        raise SelectionError(f"saved run has no results: {root}")
    for field in ("run_id", "split", "dataset_version", "dataset_sha256"):
        if summary.get(field) != run.get(field):
            raise SelectionError(f"run and summary differ on {field}: {root}")
    return SavedRun(root, run, summary, results)


def _parameters(saved: SavedRun) -> Mapping[str, object]:
    value = saved.run.get("parameters")
    if not isinstance(value, Mapping):
        raise SelectionError(f"run parameters are invalid: {saved.root}")
    return value


def _common_identity(saved: SavedRun) -> tuple[object, ...]:
    return tuple(
        saved.run.get(field)
        for field in (
            "split",
            "dataset_version",
            "dataset_sha256",
            "chunks_sha256",
            "dense_index_manifest_sha256",
        )
    )


def _hybrid_identity(saved: SavedRun) -> tuple[object, ...]:
    return (
        saved.run.get("s5_config_sha256"),
        saved.run.get("bm25_index_manifest_sha256"),
    )


def _reranker_identity(saved: SavedRun) -> tuple[object, ...]:
    return tuple(
        saved.run.get(field)
        for field in (
            "s6_config_sha256",
            "reranker_model_manifest_sha256",
            "reranker_tokenizer_sha256",
            "reranker_model_name",
            "reranker_model_revision",
            "reranker_max_length",
        )
    )


def _metric_summary(saved: SavedRun) -> tuple[float, ...]:
    retrieval = saved.summary.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise SelectionError(f"summary retrieval metrics are invalid: {saved.root}")
    values: list[float] = []
    for metric in _METRICS:
        try:
            value = float(retrieval[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise SelectionError(f"summary is missing {metric}: {saved.root}") from error
        if not math.isfinite(value):
            raise SelectionError(f"summary metric is non-finite: {saved.root}")
        values.append(value)
    return tuple(values)


def _rerank_p95(saved: SavedRun) -> float:
    try:
        operations = saved.summary["operations"]
        latency = operations["latency_ms"]  # type: ignore[index]
        reranking = latency["reranking"]  # type: ignore[index]
        value = float(reranking["p95"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as error:
        raise SelectionError(f"summary is missing reranking p95: {saved.root}") from error
    if not math.isfinite(value) or value < 0:
        raise SelectionError(f"reranking p95 is invalid: {saved.root}")
    return value


def _per_question(saved: SavedRun, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for question_id, row in saved.results.items():
        if row.get("answerable") is not True:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or metric not in metrics:
            raise SelectionError(f"result is missing {metric}: {saved.root}:{question_id}")
        value = float(metrics[metric])
        if not math.isfinite(value):
            raise SelectionError(f"result metric is non-finite: {saved.root}")
        values[question_id] = value
    if not values:
        raise SelectionError(f"run has no answerable metrics: {saved.root}")
    return values


def select_saved_validation(
    m1_root: Path,
    b1_root: Path,
    m2_roots: Sequence[Path],
    config: EvaluationConfig,
) -> dict[str, object]:
    m1 = _load(m1_root)
    b1 = _load(b1_root)
    candidates = [_load(path) for path in m2_roots]
    if m1.run.get("method") != "m1" or b1.run.get("method") != "b1":
        raise SelectionError("baseline methods must be m1 and b1")
    if len(candidates) != 3 or any(item.run.get("method") != "m2" for item in candidates):
        raise SelectionError("selection requires exactly three M2 runs")
    all_runs = [m1, b1, *candidates]
    if any(item.run.get("split") != "validation" for item in all_runs):
        raise SelectionError("selection may only read Validation runs; Test is locked")
    common = {_common_identity(item) for item in all_runs}
    if len(common) != 1:
        raise SelectionError("saved runs have different dataset or Dense identities")
    identity = next(iter(common))
    if identity[1] != config.dataset_version or identity[3] != config.chunks_sha256:
        raise SelectionError("saved run identity differs from the evaluation config")
    if any(_hybrid_identity(item) != _hybrid_identity(m1) for item in candidates):
        raise SelectionError("M1 and M2 have different hybrid identities")
    if len({_reranker_identity(item) for item in candidates}) != 1:
        raise SelectionError("M2 candidates have different reranker identities")
    question_sets = {tuple(sorted(item.results)) for item in all_runs}
    if len(question_sets) != 1:
        raise SelectionError("saved runs do not contain the same question IDs")

    by_depth: dict[int, SavedRun] = {}
    for candidate in candidates:
        parameters = _parameters(candidate)
        depth = parameters.get("rerank_depth")
        if type(depth) is not int or depth not in _DEPTHS:
            raise SelectionError(f"candidate rerank_depth is invalid: {candidate.root}")
        if parameters.get("rerank_enabled") is not True:
            raise SelectionError(f"candidate reranking is disabled: {candidate.root}")
        if depth in by_depth:
            raise SelectionError(f"duplicate M2 depth candidate: {depth}")
        by_depth[depth] = candidate
    if set(by_depth) != _DEPTHS:
        raise SelectionError("M2 candidate set does not match the pre-registered depths")

    def selection_key(item: tuple[int, SavedRun]) -> tuple[float, ...]:
        depth, saved = item
        return (*_metric_summary(saved), -_rerank_p95(saved), -depth)

    selected_depth, selected = max(by_depth.items(), key=selection_key)
    comparisons: dict[str, object] = {}
    for name, baseline in (("m2_minus_m1", m1), ("m2_minus_b1", b1)):
        comparisons[name] = {
            metric: paired_bootstrap(
                _per_question(baseline, metric),
                _per_question(selected, metric),
                config.bootstrap_seed,
                config.bootstrap_samples,
                config.confidence_level,
            )
            for metric in _METRICS
        }
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "dataset_version": identity[1],
        "dataset_sha256": identity[2],
        "chunks_sha256": identity[3],
        "candidate_depths": sorted(by_depth),
        "selection_metrics": list(_METRICS),
        "selected": {
            "run_id": selected.run.get("run_id"),
            "parameters": {"rerank_depth": selected_depth},
            "retrieval": selected.summary.get("retrieval"),
            "operations": selected.summary.get("operations"),
        },
        "candidates": {
            str(depth): {
                "run_id": saved.run.get("run_id"),
                "retrieval": saved.summary.get("retrieval"),
                "operations": saved.summary.get("operations"),
            }
            for depth, saved in sorted(by_depth.items())
        },
        "baselines": {
            "m1_run_id": m1.run.get("run_id"),
            "b1_run_id": b1.run.get("run_id"),
        },
        "reranker_identity": list(_reranker_identity(selected)),
        "comparisons": comparisons,
    }


def write_selection(path: Path, payload: Mapping[str, object]) -> None:
    ensure_directory(path.parent)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--m2", type=Path, action="append", required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = select_saved_validation(
        arguments.m1,
        arguments.b1,
        arguments.m2,
        load_evaluation_config(arguments.eval_config),
    )
    write_selection(arguments.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
