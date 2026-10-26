from __future__ import annotations

import json
from pathlib import Path

import pytest

from docguard.evaluation.config import EvaluationConfig
from docguard.evaluation.s6_selection import SelectionError, select_saved_validation


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
        bootstrap_seed=20261014,
        bootstrap_samples=1000,
        confidence_level=0.95,
        chunking_version="s2-chunk-v1",
        corpus_manifest_sha256="0" * 64,
        chunks_sha256="1" * 64,
    )


def _write_run(
    root: Path,
    *,
    method: str,
    metrics: tuple[float, float, float, float],
    depth: int | None = None,
    rerank_p95: float | None = None,
    split: str = "validation",
    dataset_sha256: str = "2" * 64,
    fail: bool = False,
) -> Path:
    root.mkdir(parents=True)
    parameters: dict[str, object] = {
        "top_k": 10,
        "dense_enabled": True,
        "bm25_enabled": method != "b1",
        "dense_depth": 50,
        "bm25_depth": 20,
        "rrf_k": 20,
    }
    if method == "m2":
        parameters.update({"rerank_enabled": True, "rerank_depth": depth})
    run: dict[str, object] = {
        "run_id": f"{method}-{depth or 0}",
        "split": split,
        "dataset_version": "s4-golden-v1",
        "dataset_sha256": dataset_sha256,
        "chunks_sha256": "1" * 64,
        "s5_config_sha256": "3" * 64 if method != "b1" else None,
        "dense_index_manifest_sha256": "4" * 64,
        "bm25_index_manifest_sha256": "5" * 64 if method != "b1" else None,
        "method": method,
        "parameters": parameters,
    }
    if method == "m2":
        run.update(
            {
                "s6_config_sha256": "6" * 64,
                "reranker_model_manifest_sha256": "7" * 64,
                "reranker_tokenizer_sha256": "8" * 64,
                "reranker_model_name": "BAAI/bge-reranker-base",
                "reranker_model_revision": "9" * 40,
                "reranker_max_length": 512,
            }
        )
    ndcg, mrr, recall10, recall5 = metrics
    metric_map = {
        "ndcg_at_10": ndcg,
        "mrr_at_10": mrr,
        "recall_at_10": recall10,
        "recall_at_5": recall5,
        "recall_at_1": 0.0,
    }
    rows = []
    for index in range(3):
        rows.append(
            {
                "question_id": f"q{index}",
                "answerable": True,
                "status": "failed" if fail and index == 0 else "success",
                "metrics": {} if fail and index == 0 else metric_map,
            }
        )
    latency = {"end_to_end": {"p50": 10.0, "p95": 20.0}}
    if method == "m2":
        latency["reranking"] = {
            "p50": float(rerank_p95) / 2,
            "p95": float(rerank_p95),
        }
    summary = {
        "run_id": run["run_id"],
        "split": split,
        "dataset_version": run["dataset_version"],
        "dataset_sha256": dataset_sha256,
        "retrieval": {"evaluated_answerable_count": 3, **metric_map},
        "operations": {
            "success_count": 2 if fail else 3,
            "failed_count": 1 if fail else 0,
            "latency_ms": latency,
        },
    }
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return root


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    m1 = _write_run(tmp_path / "m1", method="m1", metrics=(0.4, 0.4, 0.7, 0.6))
    b1 = _write_run(tmp_path / "b1", method="b1", metrics=(0.35, 0.35, 0.6, 0.5))
    m2 = [
        _write_run(
            tmp_path / f"m2-{depth}",
            method="m2",
            depth=depth,
            metrics=metrics,
            rerank_p95=latency,
        )
        for depth, metrics, latency in (
            (10, (0.42, 0.42, 0.7, 0.6), 5.0),
            (20, (0.50, 0.48, 0.8, 0.7), 9.0),
            (50, (0.48, 0.50, 0.8, 0.7), 15.0),
        )
    ]
    return m1, b1, m2


def test_s6_selection_chooses_pre_registered_quality_order_and_compares_baselines(
    tmp_path: Path,
) -> None:
    m1, b1, m2 = _fixture(tmp_path)

    result = select_saved_validation(m1, b1, m2, _config())

    assert result["selection_split"] == "validation"
    assert result["selected"]["parameters"]["rerank_depth"] == 20
    assert result["selected"]["retrieval"]["ndcg_at_10"] == 0.50
    assert set(result["comparisons"]) == {"m2_minus_m1", "m2_minus_b1"}
    assert result["candidate_depths"] == [10, 20, 50]


def test_s6_selection_breaks_quality_tie_by_latency_then_depth(tmp_path: Path) -> None:
    m1, b1, _ = _fixture(tmp_path)
    tied = [
        _write_run(
            tmp_path / f"tie-{depth}",
            method="m2",
            depth=depth,
            metrics=(0.5, 0.5, 0.8, 0.7),
            rerank_p95=latency,
        )
        for depth, latency in ((10, 8.0), (20, 7.0), (50, 7.0))
    ]

    result = select_saved_validation(m1, b1, tied, _config())

    assert result["selected"]["parameters"]["rerank_depth"] == 20


@pytest.mark.parametrize("problem", ["missing", "test", "identity", "failed"])
def test_s6_selection_rejects_incomplete_or_leaking_runs(
    tmp_path: Path, problem: str
) -> None:
    m1, b1, m2 = _fixture(tmp_path)
    if problem == "missing":
        m2.pop()
    elif problem == "test":
        m2[0] = _write_run(
            tmp_path / "test-run",
            method="m2",
            depth=10,
            metrics=(0.4, 0.4, 0.7, 0.6),
            rerank_p95=5.0,
            split="test",
        )
    elif problem == "identity":
        m2[0] = _write_run(
            tmp_path / "wrong-id",
            method="m2",
            depth=10,
            metrics=(0.4, 0.4, 0.7, 0.6),
            rerank_p95=5.0,
            dataset_sha256="f" * 64,
        )
    else:
        m2[0] = _write_run(
            tmp_path / "failed-run",
            method="m2",
            depth=10,
            metrics=(0.4, 0.4, 0.7, 0.6),
            rerank_p95=5.0,
            fail=True,
        )

    with pytest.raises(SelectionError):
        select_saved_validation(m1, b1, m2, _config())
