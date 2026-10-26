from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docguard.evaluation.s12_execution import (
    S12ExecutionError,
    begin_method,
    complete_method,
    validate_runtime_status,
    validate_seal,
)

ORDER = ("b0", "b1", "m1")


def test_seal_must_bind_current_commit_and_unviewed_results(tmp_path: Path) -> None:
    seal = {
        "status": "sealed",
        "protocol_version": "s12-final-v1",
        "git_commit": "abc",
        "protocol_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "question_count": 60,
        "results_viewed": False,
        "max_retries": 0,
        "existing_record_policy": "skip_any_record",
        "method_order": list(ORDER),
    }
    validate_seal(seal, "abc", "a" * 64, "b" * 64, ORDER)

    for field, value in (
        ("git_commit", "different"),
        ("results_viewed", True),
        ("status", "invalidated_before_test_execution"),
    ):
        changed = dict(seal)
        changed[field] = value
        with pytest.raises(S12ExecutionError):
            validate_seal(changed, "abc", "a" * 64, "b" * 64, ORDER)


def test_ledger_enforces_order_and_only_resumes_running_method() -> None:
    started = begin_method(None, "b0", ORDER, "commit")
    assert started["entries"] == [{"method": "b0", "status": "running"}]

    assert begin_method(started, "b0", ORDER, "commit") == started
    with pytest.raises(S12ExecutionError, match="running"):
        begin_method(started, "b1", ORDER, "commit")

    completed = {
        **started,
        "entries": [{"method": "b0", "status": "completed"}],
    }
    next_run = begin_method(completed, "b1", ORDER, "commit")
    assert next_run["entries"][-1] == {"method": "b1", "status": "running"}
    with pytest.raises(S12ExecutionError, match="order"):
        begin_method(completed, "m1", ORDER, "commit")


def test_ledger_rejects_commit_change_and_duplicate_completed_method() -> None:
    ledger = {
        "schema_version": 1,
        "protocol_version": "s12-final-v1",
        "git_commit": "commit",
        "entries": [{"method": "b0", "status": "completed"}],
    }
    with pytest.raises(S12ExecutionError, match="commit"):
        begin_method(ledger, "b1", ORDER, "other")
    with pytest.raises(S12ExecutionError, match="completed"):
        begin_method(ledger, "b0", ORDER, "commit")


def test_runtime_status_allows_only_sealed_outputs() -> None:
    validate_runtime_status(
        "?? reports/s12_final/preflight.json\n"
        "?? reports/s12_final/execution-ledger.json\n"
        "?? reports/s12_final/raw/b0/results.jsonl\n"
    )
    with pytest.raises(S12ExecutionError, match="unexpected working tree"):
        validate_runtime_status(" M src/docguard/evaluation/final_cli.py\n")


def test_completed_ledger_binds_exact_sixty_row_results(tmp_path: Path) -> None:
    output = tmp_path / "b0"
    output.mkdir()
    rows = [
        {
            "question_id": f"q-{index:02d}",
            "status": "failed" if index == 0 else "success",
        }
        for index in range(60)
    ]
    results = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    (output / "results.jsonl").write_text(results, encoding="utf-8")
    summary = {
        "results_sha256": hashlib.sha256(
            (output / "results.jsonl").read_bytes()
        ).hexdigest(),
        "operations": {"failed_count": 1},
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    ledger = begin_method(None, "b0", ORDER, "commit")

    completed = complete_method(ledger, "b0", output)

    entry = completed["entries"][0]
    assert entry["status"] == "completed"
    assert entry["success_count"] == 59
    assert entry["failed_count"] == 1
