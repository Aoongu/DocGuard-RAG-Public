from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "s8_refusal"


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_s8_reports_are_complete_and_test_remains_locked() -> None:
    assert (ROOT / "scripts" / "verify_s8.ps1").is_file()
    manifest = _object(REPORT / "manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["refusal_version"] == "s8-refusal-v1"
    assert manifest["test_execution_status"] == "locked_until_s12"
    files = manifest["files"]
    assert isinstance(files, dict)
    for relative, expected_hash in files.items():
        path = REPORT / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        assert "test" not in Path(relative).parts

    validation = _object(REPORT / "validation_selected" / "summary.json")
    development = _object(REPORT / "development_diagnostic" / "summary.json")
    selection = _object(REPORT / "validation_selected" / "selection.json")
    assert validation["result_status"] == "selection_result"
    assert development["result_status"] == "diagnostic_replay"
    assert validation["question_count"] == development["question_count"] == 60
    assert len(selection["candidates"]) == 1680
    assert selection["selected"]["thresholds"] == validation["selected_thresholds"]
    assert development["selected_thresholds"] == validation["selected_thresholds"]
    for directory in ("validation_selected", "development_diagnostic"):
        results = (REPORT / directory / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(results) == 60
        assert len({json.loads(line)["question_id"] for line in results}) == 60
        assert (REPORT / directory / "failures.jsonl").read_text(encoding="utf-8") == ""
