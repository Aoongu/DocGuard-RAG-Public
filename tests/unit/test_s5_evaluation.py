from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docguard.evaluation import cli as evaluation_cli
from docguard.evaluation.config import EvaluationConfig
from docguard.evaluation.models import EvaluationQuestion
from docguard.evaluation.runner import TestSplitLockedError, run_evaluation
from docguard.retrieval.hybrid import HybridSearchHit


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
        "run_id": "s5-m1-unit",
        "git_commit": "f" * 40,
        "split": split,
        "dataset_version": "s4-golden-v1",
        "dataset_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "corpus_manifest_sha256": "0" * 64,
        "chunks_sha256": "1" * 64,
        "chunking_version": "s2-chunk-v1",
        "method": "m1",
        "s5_config_sha256": "4" * 64,
        "dense_index_manifest_sha256": "5" * 64,
        "bm25_index_manifest_sha256": "6" * 64,
        "model_name": "fake",
        "model_revision": "7" * 40,
        "parameters": {
            "top_k": 10,
            "dense_enabled": True,
            "bm25_enabled": True,
            "dense_depth": 20,
            "bm25_depth": 50,
            "rrf_k": 60,
        },
    }


class HybridFake:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def search(self, query: str, version: str | None, top_k: int, **kwargs):
        self.calls.append(kwargs)
        return [
            HybridSearchHit(
                rank=1,
                chunk_id="chk_" + "a" * 64,
                score=0.08,
                rrf_score=0.08,
                sources=("dense", "bm25"),
                dense_rank=2,
                dense_score=0.7,
                bm25_rank=1,
                bm25_score=4.2,
                postgresql_version="17.11",
                topic="installation_configuration",
                section_path=("WAL",),
                canonical_url="https://www.postgresql.org/docs/17/runtime-config-wal.html",
                anchor="GUC-WAL-LEVEL",
                text="wal_level controls WAL information",
            )
        ]


def test_s5_runner_saves_method_contract_and_route_provenance(tmp_path: Path) -> None:
    retriever = HybridFake()
    output = tmp_path / "run"
    asyncio.run(
        run_evaluation([_question()], output, retriever, _config(), _metadata())
    )

    assert retriever.calls == [
        {"method": "m1", "dense_depth": 20, "bm25_depth": 50, "rrf_k": 60}
    ]
    assert json.loads((output / "run.json").read_text(encoding="utf-8")) == _metadata()
    row = json.loads((output / "results.jsonl").read_text(encoding="utf-8"))
    assert row["hits"][0] == {
        "rank": 1,
        "chunk_id": "chk_" + "a" * 64,
        "score": 0.08,
        "rrf_score": 0.08,
        "sources": ["dense", "bm25"],
        "dense_rank": 2,
        "dense_score": 0.7,
        "bm25_rank": 1,
        "bm25_score": 4.2,
    }


def test_s5_test_lock_precedes_retriever_call(tmp_path: Path) -> None:
    retriever = HybridFake()
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
    assert not (tmp_path / "run").exists()


def test_cli_test_lock_precedes_retriever_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory_calls: list[str] = []
    monkeypatch.setattr(evaluation_cli, "load_questions", lambda path: [_question("test")])
    monkeypatch.setattr(evaluation_cli, "validate_split_collection", lambda questions: None)
    monkeypatch.setattr(
        evaluation_cli,
        "create_default_retriever",
        lambda **kwargs: factory_calls.append("dense"),
    )

    with pytest.raises(TestSplitLockedError, match="S12"):
        asyncio.run(evaluation_cli._run(SimpleNamespace(dataset=tmp_path / "test.jsonl")))
    assert factory_calls == []
