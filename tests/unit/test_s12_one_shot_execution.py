from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from docguard.evaluation.config import load_evaluation_config
from docguard.evaluation.dataset import load_questions
from docguard.evaluation.final_protocol import (
    authorize_test_run,
    load_final_protocol,
)
from docguard.evaluation.runner import RunContractError, run_evaluation

ROOT = Path(__file__).parents[2]
PROTOCOL = load_final_protocol(ROOT / "configs/evaluation/s12-final-v1.json")
EVALUATION_CONFIG = load_evaluation_config(
    ROOT / "configs/evaluation/s4-eval-v1.json"
)


class FailingRetriever:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, *args: object, **kwargs: object) -> list[object]:
        self.calls += 1
        raise RuntimeError("record this failure once")


def _authorized_root(tmp_path: Path, method: str = "b1"):
    target = tmp_path / PROTOCOL.dataset.test_path
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / PROTOCOL.dataset.test_path).read_bytes())
    return authorize_test_run(tmp_path, PROTOCOL, method)


def _metadata(run_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "split": "test",
        "dataset_version": EVALUATION_CONFIG.dataset_version,
        "dataset_sha256": PROTOCOL.dataset.test_sha256,
        "config_sha256": "a" * 64,
        "chunking_version": EVALUATION_CONFIG.chunking_version,
        "corpus_manifest_sha256": EVALUATION_CONFIG.corpus_manifest_sha256,
        "chunks_sha256": EVALUATION_CONFIG.chunks_sha256,
        "method": "b1",
        "s5_config_sha256": "b" * 64,
        "dense_index_manifest_sha256": "c" * 64,
        "bm25_index_manifest_sha256": None,
        "parameters": {"dense_enabled": True, "bm25_enabled": False},
    }


def test_authorized_test_failure_is_recorded_once_and_never_retried(
    tmp_path: Path,
) -> None:
    authorization = _authorized_root(tmp_path)
    questions = load_questions(ROOT / PROTOCOL.dataset.test_path)
    retriever = FailingRetriever()
    arguments = (
        questions,
        authorization.output_dir,
        retriever,
        EVALUATION_CONFIG,
        _metadata(authorization.run_id),
    )

    first = asyncio.run(
        run_evaluation(*arguments, test_authorization=authorization)
    )
    second = asyncio.run(
        run_evaluation(*arguments, test_authorization=authorization)
    )

    assert first == second
    assert retriever.calls == 60
    rows = [
        json.loads(line)
        for line in (authorization.output_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 60
    assert all(row["status"] == "failed" for row in rows)


def test_authorized_test_rejects_wrong_output_before_retrieval(tmp_path: Path) -> None:
    authorization = _authorized_root(tmp_path)
    questions = load_questions(ROOT / PROTOCOL.dataset.test_path)
    retriever = FailingRetriever()

    with pytest.raises(RunContractError, match="authorization"):
        asyncio.run(
            run_evaluation(
                questions,
                tmp_path / "wrong",
                retriever,
                EVALUATION_CONFIG,
                _metadata(authorization.run_id),
                test_authorization=authorization,
            )
        )

    assert retriever.calls == 0
    assert not (tmp_path / "wrong").exists()


def test_authorized_test_requires_the_complete_frozen_question_set(
    tmp_path: Path,
) -> None:
    authorization = _authorized_root(tmp_path)
    questions = load_questions(ROOT / PROTOCOL.dataset.test_path)
    retriever = FailingRetriever()

    with pytest.raises(RunContractError, match="complete frozen Test"):
        asyncio.run(
            run_evaluation(
                questions[:-1],
                authorization.output_dir,
                retriever,
                EVALUATION_CONFIG,
                _metadata(authorization.run_id),
                test_authorization=authorization,
            )
        )

    assert retriever.calls == 0
    assert not authorization.output_dir.exists()


def test_authorization_rejects_tampered_test_dataset(tmp_path: Path) -> None:
    target = tmp_path / PROTOCOL.dataset.test_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="Test dataset hash"):
        authorize_test_run(tmp_path, PROTOCOL, "b1")
