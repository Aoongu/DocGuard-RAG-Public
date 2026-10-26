from __future__ import annotations

import json
from pathlib import Path

import pytest

from docguard.refusal.config import load_refusal_config
from docguard.refusal.models import Action, ReasonCode

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "refusal" / "s8-refusal-v1.json"


def test_frozen_s8_config() -> None:
    config = load_refusal_config(CONFIG)
    assert config.refusal_version == "s8-refusal-v1"
    assert config.supported_versions == ("17.11", "18.6")
    assert config.threshold_grid.top1 == (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0)
    assert config.threshold_grid.coverage == (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
    assert config.selection_objective == (
        "macro_f1",
        "false_accept_count",
        "refusal_precision",
        "answerable_coverage",
        "threshold_lexicographic",
    )
    assert config.test_execution_status == "locked_until_s12"
    assert Action.CLARIFY.value == "clarify"
    assert ReasonCode.VERSION_NOT_AVAILABLE.value == "VERSION_NOT_AVAILABLE"


@pytest.mark.parametrize("mutation", ["unknown", "missing", "test_unlock"])
def test_config_fails_closed(tmp_path: Path, mutation: str) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    if mutation == "unknown":
        payload["surprise"] = True
    elif mutation == "missing":
        del payload["selection_objective"]
    else:
        payload["test_execution_status"] = "unlocked"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_refusal_config(path)
