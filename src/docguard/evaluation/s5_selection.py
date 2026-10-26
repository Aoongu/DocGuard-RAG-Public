"""Saved-result-only S5 Validation selection and paired comparisons."""

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
_GRID = frozenset(
    (rrf_k, dense_depth, bm25_depth)
    for rrf_k in (20, 60)
    for dense_depth in (20, 50)
    for bm25_depth in (20, 50)
)


class SelectionError(RuntimeError):
    """Raised when saved S5 runs are incomplete or incomparable."""


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
        validate_regular_file(root / "results.jsonl")
        results: dict[str, dict[str, object]] = {}
        for line_number, line in enumerate(
            (root / "results.jsonl").read_text(encoding="utf-8").splitlines(), 1
        ):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
                raise SelectionError(f"invalid result row at {root}:{line_number}")
            question_id = row["question_id"]
            if question_id in results:
                raise SelectionError(f"duplicate result question ID: {question_id}")
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
    parameters = saved.run.get("parameters")
    if not isinstance(parameters, Mapping):
        raise SelectionError(f"run parameters are invalid: {saved.root}")
    return parameters


def _identity(saved: SavedRun) -> tuple[object, ...]:
    return tuple(
        saved.run.get(field)
        for field in (
            "split",
            "dataset_version",
            "dataset_sha256",
            "chunks_sha256",
            "s5_config_sha256",
        )
    )


def _metric_summary(saved: SavedRun) -> tuple[float, ...]:
    retrieval = saved.summary.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise SelectionError(f"summary retrieval metrics are invalid: {saved.root}")
    values = []
    for metric in _METRICS:
        try:
            value = float(retrieval[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise SelectionError(f"summary is missing {metric}: {saved.root}") from error
        if not math.isfinite(value):
            raise SelectionError(f"summary metric is non-finite: {saved.root}")
        values.append(value)
    return tuple(values)


def _per_question(saved: SavedRun, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for question_id, row in saved.results.items():
        if row.get("status") != "success" or row.get("answerable") is not True:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or metric not in metrics:
            raise SelectionError(f"result is missing {metric}: {saved.root}:{question_id}")
        value = float(metrics[metric])
        if not math.isfinite(value):
            raise SelectionError(f"result metric is non-finite: {saved.root}:{question_id}")
        values[question_id] = value
    if not values:
        raise SelectionError(f"run has no answerable metrics: {saved.root}")
    return values


def select_saved_validation(
    b0_root: Path,
    b1_root: Path,
    m1_roots: Sequence[Path],
    config: EvaluationConfig,
) -> dict[str, object]:
    b0 = _load(b0_root)
    b1 = _load(b1_root)
    candidates = [_load(path) for path in m1_roots]
    if b0.run.get("method") != "b0" or b1.run.get("method") != "b1":
        raise SelectionError("baseline methods must be b0 and b1")
    if len(candidates) != 8 or any(item.run.get("method") != "m1" for item in candidates):
        raise SelectionError("selection requires exactly eight M1 runs")
    all_runs = [b0, b1, *candidates]
    if any(item.run.get("split") != "validation" for item in all_runs):
        raise SelectionError("selection may only read Validation runs; Test is locked")
    identities = {_identity(item) for item in all_runs}
    if len(identities) != 1:
        raise SelectionError("saved runs have different dataset or artifact identities")
    identity = next(iter(identities))
    if identity[1] != config.dataset_version or identity[3] != config.chunks_sha256:
        raise SelectionError("saved run identity differs from the evaluation config")
    question_sets = {tuple(sorted(item.results)) for item in all_runs}
    if len(question_sets) != 1:
        raise SelectionError("saved runs do not contain the same question IDs")

    by_grid: dict[tuple[int, int, int], SavedRun] = {}
    for candidate in candidates:
        parameters = _parameters(candidate)
        values = (
            parameters.get("rrf_k"),
            parameters.get("dense_depth"),
            parameters.get("bm25_depth"),
        )
        if any(type(value) is not int for value in values):
            raise SelectionError(f"candidate grid values are invalid: {candidate.root}")
        grid = (int(values[0]), int(values[1]), int(values[2]))
        if grid in by_grid:
            raise SelectionError(f"duplicate M1 grid candidate: {grid}")
        by_grid[grid] = candidate
    if set(by_grid) != _GRID:
        raise SelectionError("M1 candidate set does not match the pre-registered grid")

    def selection_key(item: tuple[tuple[int, int, int], SavedRun]) -> tuple[float, ...]:
        (rrf_k, dense_depth, bm25_depth), saved = item
        return (
            *_metric_summary(saved),
            -(dense_depth + bm25_depth),
            -rrf_k,
            -dense_depth,
            -bm25_depth,
        )

    selected_grid, selected = max(by_grid.items(), key=selection_key)
    comparisons: dict[str, object] = {}
    for name, baseline in (("m1_minus_b0", b0), ("m1_minus_b1", b1)):
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
    rrf_k, dense_depth, bm25_depth = selected_grid
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "dataset_version": identity[1],
        "dataset_sha256": identity[2],
        "chunks_sha256": identity[3],
        "s5_config_sha256": identity[4],
        "selection_metrics": list(_METRICS),
        "selected": {
            "run_id": selected.run.get("run_id"),
            "parameters": {
                "dense_depth": dense_depth,
                "bm25_depth": bm25_depth,
                "rrf_k": rrf_k,
            },
            "retrieval": selected.summary.get("retrieval"),
        },
        "baselines": {
            "b0_run_id": b0.run.get("run_id"),
            "b1_run_id": b1.run.get("run_id"),
        },
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
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--m1", type=Path, action="append", required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = select_saved_validation(
        arguments.b0,
        arguments.b1,
        arguments.m1,
        load_evaluation_config(arguments.eval_config),
    )
    write_selection(arguments.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
