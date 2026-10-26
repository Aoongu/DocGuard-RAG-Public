from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from docguard.evaluation.refusal_metrics import (
    MetricRecord,
    aggregate_refusal_metrics,
    expected_action,
)
from docguard.evaluation.refusal_runner import CalibrationExample, calibrate_thresholds
from docguard.refusal.config import load_refusal_config
from docguard.refusal.models import Action, RefusalFeatures
from docguard.refusal.versioning import VersionRequest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_refusal_config(ROOT / "configs" / "refusal" / "s8-refusal-v1.json")
BASE = RefusalFeatures(
    top1_rerank_score=1.0,
    top1_top2_margin=0.5,
    top3_mean_rerank_score=0.0,
    evidence_term_coverage=0.75,
    requested_version=None,
    requested_version_status="none",
    top_evidence_versions=("17.11",),
    best_score_by_version=(("17.11", 1.0),),
    paired_version_conflict=False,
    postgresql_domain_signal=True,
)
NONE = VersionRequest(None, "none", (), False)


def test_expected_action_maps_historical_labels() -> None:
    assert expected_action(True, None) is Action.ANSWER
    assert expected_action(False, "NO_EVIDENCE") is Action.REFUSE
    assert expected_action(False, "UNSUPPORTED_VERSION") is Action.REFUSE
    assert expected_action(False, "VERSION_REQUIRED") is Action.CLARIFY
    with pytest.raises(ValueError):
        expected_action(False, "UNKNOWN")


def test_metrics_report_refusal_errors_auc_and_risk_coverage() -> None:
    records = [
        MetricRecord("a", Action.ANSWER, Action.ANSWER, 0.1),
        MetricRecord("b", Action.REFUSE, Action.ANSWER, 0.2),
        MetricRecord("c", Action.ANSWER, Action.REFUSE, 0.8),
        MetricRecord("d", Action.CLARIFY, Action.CLARIFY, 0.9),
    ]
    result = aggregate_refusal_metrics(records)
    assert result.n == 4
    assert result.refusal_precision == pytest.approx(0.5)
    assert result.refusal_recall == pytest.approx(0.5)
    assert result.refusal_f1 == pytest.approx(0.5)
    assert result.false_accept_count == 1
    assert result.false_reject_count == 1
    assert result.roc_auc == pytest.approx(0.75)
    assert result.confusion_matrix["refuse"]["answer"] == 1
    assert result.risk_coverage[-1].coverage == 1.0
    assert result.risk_coverage[-1].selective_risk == pytest.approx(0.5)


def test_metrics_use_null_for_undefined_denominators() -> None:
    result = aggregate_refusal_metrics(
        [MetricRecord("a", Action.ANSWER, Action.ANSWER, 0.1)]
    )
    assert result.refusal_precision is None
    assert result.refusal_recall is None
    assert result.refusal_f1 is None
    assert result.roc_auc is None


def test_grid_selection_saves_all_candidates_and_is_deterministic() -> None:
    examples = [
        CalibrationExample("answer", BASE, NONE, Action.ANSWER),
        CalibrationExample(
            "boundary",
            replace(
                BASE,
                top1_rerank_score=-5.5,
                top1_top2_margin=0.0,
                top3_mean_rerank_score=-5.5,
                evidence_term_coverage=0.0,
                postgresql_domain_signal=False,
            ),
            NONE,
            Action.REFUSE,
        ),
    ]
    first = calibrate_thresholds(examples, CONFIG)
    second = calibrate_thresholds(list(reversed(examples)), CONFIG)
    assert len(first.candidates) == 8 * 7 * 6 * 5
    assert first.selected.thresholds == second.selected.thresholds
    assert first.selected.metrics.macro_f1 == 0.5
    assert [item.thresholds for item in first.candidates] == sorted(
        item.thresholds for item in first.candidates
    )
