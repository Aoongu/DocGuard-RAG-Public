import hashlib
import json
import subprocess
import sys
from pathlib import Path

from docguard.discovery.store import TaskStore


def test_trace_export_is_read_only_and_includes_the_complete_event_ledger(tmp_path):
    db = tmp_path / "state.sqlite3"
    store = TaskStore(db)
    task = store.create(
        {"goal": "pandas 官方文档", "product": "pandas", "version": None, "urls": []}
    )
    for i in range(505):
        store.mutate(task["task_id"], lambda row: None, "fixture_event", {"i": i})
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    output = tmp_path / "log.json"
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2] / "scripts/export_discovery_log.py"
            ),
            "--database",
            str(db),
            "--task-id",
            task["task_id"],
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["task"]["goal"] == "pandas 官方文档"
    assert len(payload["events"]) == 506
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
