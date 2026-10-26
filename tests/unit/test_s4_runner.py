import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docguard.evaluation.config import EvaluationConfig
from docguard.evaluation.models import EvaluationQuestion
from docguard.evaluation.runner import (
    RunContractError,
    TestSplitLockedError,
    run_evaluation,
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
        bootstrap_samples=10000,
        confidence_level=0.95,
        chunking_version="s2-chunk-v1",
        corpus_manifest_sha256="0" * 64,
        chunks_sha256="1" * 64,
    )


def _question(question_id: str, split: str = "development") -> EvaluationQuestion:
    prefix = {"development": "dev", "validation": "val", "test": "test"}[split]
    return EvaluationQuestion.from_dict(
        {
            "schema_version": 1,
            "question_id": f"s4-{prefix}-{question_id}",
            "question": "What does SELECT do?",
            "requested_version": "18.6",
            "answerable": True,
            "answer_key_points": ["It retrieves rows."],
            "relevant_source_refs": [
                {
                    "version": "18.6",
                    "normalized_url": "https://www.postgresql.org/docs/18/sql-select.html",
                    "anchor": "SQL-SELECT",
                }
            ],
            "resolved_chunk_ids": ["chk_" + "a" * 64],
            "chunking_version": "s2-chunk-v1",
            "expected_reason_code": None,
            "category": "single_fact",
            "split": split,
            "source": "assistant_curated",
            "review_status": "source_verified",
            "paraphrase_group_id": f"select-{split}",
        }
    )


class FakeRetriever:
    def __init__(self, fail_query: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_query = fail_query

    async def search(self, query: str, version: str | None, top_k: int, dense_enabled: bool = True):
        self.calls.append(query)
        if query == self.fail_query:
            raise RuntimeError("controlled failure")
        return [
            SimpleNamespace(rank=1, chunk_id="chk_" + "a" * 64, score=0.9),
            SimpleNamespace(rank=2, chunk_id="chk_" + "b" * 64, score=0.8),
        ]


def _metadata(split: str = "development") -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "s4-unit-run",
        "git_commit": "f" * 40,
        "split": split,
        "dataset_version": "s4-golden-v1",
        "dataset_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "corpus_manifest_sha256": "0" * 64,
        "chunks_sha256": "1" * 64,
        "chunking_version": "s2-chunk-v1",
        "model_name": "fake",
        "model_revision": "4" * 40,
        "parameters": {"top_k": 10},
    }


def _clock():
    values = iter([0.0, 0.01, 0.012] * 10)
    return lambda: next(values)


def test_runner_writes_results_summary_and_resumes_successes(tmp_path: Path) -> None:
    questions = [_question("001"), _question("002")]
    retriever = FakeRetriever()
    output = tmp_path / "run"

    first = asyncio.run(
        run_evaluation(questions, output, retriever, _config(), _metadata(), clock=_clock())
    )
    assert first["retrieval"]["recall_at_1"] == 1.0
    assert first["operations"]["latency_ms"]["retrieval"]["p50"] == 10.0
    assert first["operations"]["usage"]["api_cost_usd"] == 0.0
    assert len(retriever.calls) == 2

    second = asyncio.run(
        run_evaluation(questions, output, retriever, _config(), _metadata(), clock=_clock())
    )
    assert second == first
    assert len(retriever.calls) == 2
    assert len((output / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_runner_isolates_per_question_failures(tmp_path: Path) -> None:
    good = _question("001")
    bad_payload = _question("002").to_dict()
    bad_payload["question"] = "fail me"
    bad = EvaluationQuestion.from_dict(bad_payload)
    output = tmp_path / "run"

    summary = asyncio.run(
        run_evaluation(
            [good, bad],
            output,
            FakeRetriever(fail_query="fail me"),
            _config(),
            _metadata(),
            clock=_clock(),
        )
    )

    assert summary["operations"]["success_count"] == 1
    assert summary["operations"]["failed_count"] == 1
    failure = json.loads((output / "failures.jsonl").read_text(encoding="utf-8"))
    assert failure["question_id"] == bad.question_id
    assert failure["error_type"] == "RuntimeError"


def test_runner_rejects_test_before_retrieval(tmp_path: Path) -> None:
    retriever = FakeRetriever()
    with pytest.raises(TestSplitLockedError, match="S12"):
        asyncio.run(
            run_evaluation(
                [_question("001", "test")],
                tmp_path / "run",
                retriever,
                _config(),
                _metadata("test"),
                clock=_clock(),
            )
        )
    assert retriever.calls == []
    assert not (tmp_path / "run").exists()


def test_runner_rejects_resume_contract_change(tmp_path: Path) -> None:
    output = tmp_path / "run"
    questions = [_question("001")]
    asyncio.run(
        run_evaluation(questions, output, FakeRetriever(), _config(), _metadata(), clock=_clock())
    )
    changed = _metadata()
    changed["dataset_sha256"] = "9" * 64
    with pytest.raises(RunContractError, match="run metadata"):
        asyncio.run(
            run_evaluation(questions, output, FakeRetriever(), _config(), changed, clock=_clock())
        )
