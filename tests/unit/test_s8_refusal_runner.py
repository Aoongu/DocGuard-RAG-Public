from __future__ import annotations

import json
from pathlib import Path

import pytest

from docguard.evaluation.refusal_runner import (
    RefusalRunStore,
    TestSplitLockedError,
    assert_non_test_input,
)


def test_test_lock_rejects_split_path_and_content(tmp_path: Path) -> None:
    safe = tmp_path / "validation.jsonl"
    safe.write_text('{"question_id":"q","split":"validation"}\n', encoding="utf-8")
    assert_non_test_input(safe, "validation")
    with pytest.raises(TestSplitLockedError):
        assert_non_test_input(safe, "test")
    test_path = tmp_path / "test.jsonl"
    test_path.write_text('{"question_id":"q","split":"validation"}\n', encoding="utf-8")
    with pytest.raises(TestSplitLockedError):
        assert_non_test_input(test_path, "validation")
    hidden = tmp_path / "hidden.jsonl"
    hidden.write_text('{"question_id":"q","split":"test"}\n', encoding="utf-8")
    with pytest.raises(TestSplitLockedError):
        assert_non_test_input(hidden, "validation")


def test_run_store_is_immutable_and_skips_any_existing_result(tmp_path: Path) -> None:
    output = tmp_path / "run"
    metadata = {"run_id": "s8-v1", "split": "validation", "status": "selection_result"}
    store = RefusalRunStore(output, metadata)
    assert store.append({"question_id": "q1", "status": "success"}) is True
    assert store.append({"question_id": "q1", "status": "failed"}) is False
    assert len((output / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    RefusalRunStore(output, metadata)
    with pytest.raises(ValueError):
        RefusalRunStore(output, {**metadata, "run_id": "changed"})
    saved = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert saved == metadata
