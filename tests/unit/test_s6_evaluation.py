from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from docguard.evaluation import cli as evaluation_cli
from docguard.evaluation.config import EvaluationConfig
from docguard.evaluation.models import EvaluationQuestion
from docguard.evaluation.runner import (
    RunContractError,
    TestSplitLockedError,
    run_evaluation,
)
from docguard.retrieval.reranker import RerankedSearchHit, RerankerError


def test_s6_cli_exposes_frozen_m2_arguments(tmp_path: Path) -> None:
    args = evaluation_cli._parser().parse_args(
        [
            "run",
            "--dataset",
            str(tmp_path / "validation.jsonl"),
            "--output",
            str(tmp_path / "output"),
            "--run-id",
            "s6-validation-d20",
            "--method",
            "m2",
            "--rerank-depth",
            "20",
        ]
    )

    assert args.method == "m2"
    assert args.rerank_depth == 20
    assert args.rerank_enabled is True
    assert args.s6_config == Path("configs/retrieval/s6-reranker-v1.json")
    assert args.reranker_model_root == Path("data/models/s6-bge-reranker-base")
    assert args.reranker_model_manifest == Path(
        "data/manifests/s6-reranker-model.json"
    )


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


def _question(split: str = "validation") -> EvaluationQuestion:
    prefix = "val" if split == "validation" else "test"
    return EvaluationQuestion.from_dict(
        {
            "schema_version": 1,
            "question_id": f"s4-{prefix}-001",
            "question": "What is wal_level?",
            "requested_version": "17.11",
            "answerable": True,
            "answer_key_points": ["Controls WAL information."],
            "relevant_source_refs": [
                {
                    "version": "17.11",
                    "normalized_url": "https://www.postgresql.org/docs/17/runtime-config-wal.html",
                    "anchor": "GUC-WAL-LEVEL",
                }
            ],
            "resolved_chunk_ids": ["chk_" + "a" * 64],
            "chunking_version": "s2-chunk-v1",
            "expected_reason_code": None,
            "category": "single_fact",
            "split": split,
            "source": "human_written",
            "review_status": "human_reviewed",
            "paraphrase_group_id": f"wal-{split}",
        }
    )


def _metadata(split: str = "validation") -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "s6-m2-unit",
        "git_commit": "f" * 40,
        "split": split,
        "dataset_version": "s4-golden-v1",
        "dataset_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "corpus_manifest_sha256": "0" * 64,
        "chunks_sha256": "1" * 64,
        "chunking_version": "s2-chunk-v1",
        "method": "m2",
        "s5_config_sha256": "4" * 64,
        "s6_config_sha256": "5" * 64,
        "dense_index_manifest_sha256": "6" * 64,
        "bm25_index_manifest_sha256": "7" * 64,
        "model_name": "dense-model",
        "model_revision": "8" * 40,
        "reranker_model_name": "BAAI/bge-reranker-base",
        "reranker_model_revision": "9" * 40,
        "reranker_model_manifest_sha256": "a" * 64,
        "reranker_tokenizer_sha256": "b" * 64,
        "reranker_max_length": 512,
        "parameters": {
            "top_k": 10,
            "dense_enabled": True,
            "bm25_enabled": True,
            "dense_depth": 50,
            "bm25_depth": 20,
            "rrf_k": 20,
            "rerank_enabled": True,
            "rerank_depth": 20,
        },
    }


class M2Fake:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail = fail
        self.timing_consumed = 0

    async def search(self, query: str, version: str | None, top_k: int, **kwargs):
        del query, version, top_k
        self.calls.append(kwargs)
        if self.fail:
            raise RerankerError("controlled model failure")
        return [
            RerankedSearchHit(
                rank=1,
                chunk_id="chk_" + "a" * 64,
                score=3.5,
                rerank_score=3.5,
                rerank_truncated=True,
                original_rrf_rank=4,
                rrf_score=0.05,
                sources=("dense", "bm25"),
                dense_rank=5,
                dense_score=0.7,
                bm25_rank=2,
                bm25_score=4.2,
                postgresql_version="17.11",
                topic="installation_configuration",
                section_path=("WAL",),
                canonical_url="https://www.postgresql.org/docs/17/runtime-config-wal.html",
                anchor="GUC-WAL-LEVEL",
                text="wal_level controls WAL information",
            )
        ]

    def consume_timing_ms(self) -> dict[str, float]:
        self.timing_consumed += 1
        return {"retrieval": 4.0, "reranking": 6.0}


def test_s6_runner_saves_m2_contract_provenance_and_segmented_latency(
    tmp_path: Path,
) -> None:
    retriever = M2Fake()
    ticks = iter((0.0, 0.010, 0.011))
    output = tmp_path / "run"

    summary = asyncio.run(
        run_evaluation(
            [_question()],
            output,
            retriever,
            _config(),
            _metadata(),
            clock=lambda: next(ticks),
        )
    )

    assert retriever.calls == [
        {"method": "m2", "rerank_enabled": True, "rerank_depth": 20}
    ]
    assert retriever.timing_consumed == 1
    row = json.loads((output / "results.jsonl").read_text(encoding="utf-8"))
    assert row["hits"][0] == {
        "rank": 1,
        "chunk_id": "chk_" + "a" * 64,
        "score": 3.5,
        "rerank_score": 3.5,
        "rerank_truncated": True,
        "original_rrf_rank": 4,
        "rrf_score": 0.05,
        "sources": ["dense", "bm25"],
        "dense_rank": 5,
        "dense_score": 0.7,
        "bm25_rank": 2,
        "bm25_score": 4.2,
    }
    assert row["latency_ms"] == {
        "retrieval": 4.0,
        "reranking": 6.0,
        "end_to_end": 11.0,
    }
    assert summary["operations"]["latency_ms"]["reranking"] == {
        "p50": 6.0,
        "p95": 6.0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("s6_config_sha256"), "s6_config_sha256"),
        (
            lambda value: value["parameters"].__setitem__("rerank_depth", 30),
            "rerank_depth",
        ),
        (
            lambda value: value.__setitem__("reranker_max_length", 256),
            "reranker_max_length",
        ),
    ],
)
def test_s6_runner_rejects_m2_identity_drift_before_search(
    tmp_path: Path, mutation, message: str
) -> None:
    metadata = _metadata()
    mutation(metadata)
    retriever = M2Fake()

    with pytest.raises(RunContractError, match=message):
        asyncio.run(
            run_evaluation(
                [_question()], tmp_path / "run", retriever, _config(), metadata
            )
        )
    assert retriever.calls == []


def test_s6_model_failure_is_saved_without_silent_success(tmp_path: Path) -> None:
    output = tmp_path / "run"
    summary = asyncio.run(
        run_evaluation(
            [_question()], output, M2Fake(fail=True), _config(), _metadata()
        )
    )

    row = json.loads((output / "results.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert row["hits"] == []
    assert row["error_type"] == "RerankerError"
    assert "controlled model failure" in row["error_message"]
    assert summary["operations"]["failed_count"] == 1


def test_s6_test_lock_precedes_m2_retriever_call(tmp_path: Path) -> None:
    retriever = M2Fake()
    with pytest.raises(TestSplitLockedError, match="S12"):
        asyncio.run(
            run_evaluation(
                [_question("test")],
                tmp_path / "run",
                retriever,
                _config(),
                _metadata("test"),
            )
        )
    assert retriever.calls == []
