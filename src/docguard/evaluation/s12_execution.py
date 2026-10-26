"""Sealed execution ledger for the one-shot S12 Test sequence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from docguard.evaluation.runner import _atomic_text, _json_line


class S12ExecutionError(RuntimeError):
    """Raised when a formal S12 run violates its sealed sequence."""


def validate_runtime_status(status: str) -> None:
    """Allow only the seal, ledger, and formal outputs after a clean preflight."""
    allowed_files = {
        "reports/s12_final/preflight.json",
        "reports/s12_final/execution-ledger.json",
    }
    for line in status.splitlines():
        path = line[3:].strip().replace("\\", "/")
        if path in allowed_files or path.startswith("reports/s12_final/raw/"):
            continue
        raise S12ExecutionError(f"unexpected working tree change after sealing: {path}")


def validate_seal(
    seal: Mapping[str, object],
    git_commit: str,
    protocol_sha256: str,
    dataset_sha256: str,
    method_order: Sequence[str],
) -> None:
    expected = {
        "status": "sealed",
        "protocol_version": "s12-final-v1",
        "git_commit": git_commit,
        "protocol_sha256": protocol_sha256,
        "dataset_sha256": dataset_sha256,
        "question_count": 60,
        "results_viewed": False,
        "max_retries": 0,
        "existing_record_policy": "skip_any_record",
        "method_order": list(method_order),
    }
    if any(seal.get(field) != value for field, value in expected.items()):
        raise S12ExecutionError("preflight seal does not match the current S12 commit")


def begin_method(
    ledger: Mapping[str, object] | None,
    method: str,
    method_order: Sequence[str],
    git_commit: str,
) -> dict[str, object]:
    if method not in method_order:
        raise S12ExecutionError("method is outside the frozen S12 order")
    if ledger is None:
        if method != method_order[0]:
            raise S12ExecutionError("method violates the frozen execution order")
        return {
            "schema_version": 1,
            "protocol_version": "s12-final-v1",
            "git_commit": git_commit,
            "entries": [{"method": method, "status": "running"}],
        }
    if ledger.get("git_commit") != git_commit:
        raise S12ExecutionError("execution ledger commit differs from the seal commit")
    entries_value = ledger.get("entries")
    if not isinstance(entries_value, list) or not all(
        isinstance(item, dict) for item in entries_value
    ):
        raise S12ExecutionError("execution ledger entries are invalid")
    entries = [dict(item) for item in entries_value]
    if entries and entries[-1].get("status") == "running":
        if entries[-1].get("method") != method:
            raise S12ExecutionError("another S12 method is still running")
        return dict(ledger)
    completed = [str(item.get("method")) for item in entries]
    if method in completed:
        raise S12ExecutionError("completed S12 methods cannot be executed again")
    expected_index = len(completed)
    if expected_index >= len(method_order) or method != method_order[expected_index]:
        raise S12ExecutionError("method violates the frozen execution order")
    return {
        **dict(ledger),
        "entries": [*entries, {"method": method, "status": "running"}],
    }


def load_ledger(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise S12ExecutionError("execution ledger must be a JSON object")
    return value


def save_ledger(path: Path, ledger: Mapping[str, object]) -> None:
    _atomic_text(path, _json_line(dict(ledger)))


def complete_method(
    ledger: Mapping[str, object], method: str, output_dir: Path
) -> dict[str, object]:
    entries_value = ledger.get("entries")
    if not isinstance(entries_value, list) or not entries_value:
        raise S12ExecutionError("execution ledger has no running method")
    entries = [dict(item) for item in entries_value if isinstance(item, dict)]
    current = entries[-1]
    if current != {"method": method, "status": "running"}:
        raise S12ExecutionError("ledger does not contain the expected running method")
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(rows) != 60 or not isinstance(summary, dict):
        raise S12ExecutionError("formal S12 run must contain exactly 60 result rows")
    result_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
    if summary.get("results_sha256") != result_hash:
        raise S12ExecutionError("summary does not bind the formal result bytes")
    failed = sum(row.get("status") == "failed" for row in rows)
    identities = sorted(
        {
            (str(row.get("model")), str(row.get("provider_fingerprint")))
            for row in rows
            if row.get("model") is not None
        }
    )
    entries[-1] = {
        "method": method,
        "status": "completed",
        "results_sha256": result_hash,
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "question_count": 60,
        "success_count": 60 - failed,
        "failed_count": failed,
        "operations": summary.get("operations"),
        "model_identities": [
            {"model": model, "provider_fingerprint": fingerprint}
            for model, fingerprint in identities
        ],
    }
    return {**dict(ledger), "entries": entries}
