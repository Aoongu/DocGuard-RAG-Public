from __future__ import annotations

import json
from pathlib import Path

import pytest

from docguard.evaluation.config import EvaluationConfig
from docguard.evaluation.s5_selection import SelectionError, select_saved_validation


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        schema_version=1,
        evaluation_version="s4-eval-v1",
        dataset_version="s4-golden-v1",
        allowed_splits=("development", "validation"),
        retrieval_cutoffs=(1, 5, 10),
        mrr_cutoff=10,
        ndcg_cutoff=10,
        retrieval_top_k=10,
        bootstrap_seed=7,
        bootstrap_samples=1000,
        confidence_level=0.95,
        chunking_version="s2-chunk-v1",
        corpus_manifest_sha256="0" * 64,
        chunks_sha256="1" * 64,
    )


def _write_run(
    root: Path,
    method: str,
    *,
    dense_depth: int = 20,
    bm25_depth: int = 20,
    rrf_k: int = 20,
    ndcg: float = 0.5,
    dataset_sha256: str = "2" * 64,
    split: str = "validation",
) -> None:
    root.mkdir()
    run = {
        "schema_version": 1,
        "run_id": root.name,
        "split": split,
        "method": method,
        "dataset_version": "s4-golden-v1",
        "dataset_sha256": dataset_sha256,
        "chunks_sha256": "1" * 64,
        "s5_config_sha256": "3" * 64,
        "parameters": {
            "top_k": 10,
            "dense_enabled": method != "b0",
            "bm25_enabled": method != "b1",
            "dense_depth": dense_depth,
            "bm25_depth": bm25_depth,
            "rrf_k": rrf_k,
        },
    }
    summary = {
        "schema_version": 1,
        "run_id": root.name,
        "split": split,
        "dataset_version": "s4-golden-v1",
        "dataset_sha256": dataset_sha256,
        "retrieval": {
            "evaluated_answerable_count": 2,
            "ndcg_at_10": ndcg,
            "mrr_at_10": 0.5,
            "recall_at_10": 0.5,
            "recall_at_5": 0.5,
        },
    }
    results = [
        {
            "question_id": question_id,
            "status": "success",
            "answerable": True,
            "metrics": {
                "ndcg_at_10": ndcg,
                "mrr_at_10": 0.5,
                "recall_at_10": 0.5,
                "recall_at_5": 0.5,
            },
        }
        for question_id in ("s4-val-001", "s4-val-002")
    ]
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8"
    )


def _grid(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    b0 = tmp_path / "b0"
    b1 = tmp_path / "b1"
    _write_run(b0, "b0")
    _write_run(b1, "b1")
    candidates = []
    for rrf_k in (20, 60):
        for dense_depth in (20, 50):
            for bm25_depth in (20, 50):
                path = tmp_path / f"m1-k{rrf_k}-d{dense_depth}-b{bm25_depth}"
                _write_run(
                    path,
                    "m1",
                    rrf_k=rrf_k,
                    dense_depth=dense_depth,
                    bm25_depth=bm25_depth,
                )
                candidates.append(path)
    return b0, b1, candidates


def test_selection_uses_metric_order_then_smaller_depth_and_rrf(tmp_path: Path) -> None:
    b0, b1, candidates = _grid(tmp_path)
    selected = select_saved_validation(b0, b1, candidates, _config())

    assert selected["selected"]["parameters"] == {
        "dense_depth": 20,
        "bm25_depth": 20,
        "rrf_k": 20,
    }
    assert set(selected["comparisons"]) == {"m1_minus_b0", "m1_minus_b1"}
    assert selected["comparisons"]["m1_minus_b0"]["ndcg_at_10"]["samples"] == 1000


@pytest.mark.parametrize("problem", ["missing", "extra", "hash", "test"])
def test_selection_rejects_incomplete_or_incomparable_runs(
    tmp_path: Path, problem: str
) -> None:
    b0, b1, candidates = _grid(tmp_path)
    if problem == "missing":
        candidates.pop()
    elif problem == "extra":
        duplicate = tmp_path / "extra"
        _write_run(duplicate, "m1")
        candidates.append(duplicate)
    elif problem == "hash":
        run_path = candidates[0] / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["dataset_sha256"] = "9" * 64
        run_path.write_text(json.dumps(run), encoding="utf-8")
    else:
        run_path = candidates[0] / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["split"] = "test"
        run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(SelectionError):
        select_saved_validation(b0, b1, candidates, _config())
