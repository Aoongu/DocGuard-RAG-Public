from __future__ import annotations

from pathlib import Path

import pytest

from docguard.engineering.config import RegressionThresholds
from docguard.engineering.regression import (
    RankedHit,
    assert_regression_thresholds,
    compare_rankings,
    evaluate_fixture,
)


def test_rankings_allow_only_score_tied_order_differences() -> None:
    local = [RankedHit("a", 0.9), RankedHit("b", 0.8), RankedHit("c", 0.8000004)]
    service = [RankedHit("a", 0.9), RankedHit("c", 0.8), RankedHit("b", 0.8000003)]
    compare_rankings(local, service, tolerance=1e-6)

    with pytest.raises(AssertionError, match="rank"):
        compare_rankings(
            local,
            [RankedHit("b", 0.8), RankedHit("c", 0.8000004), RankedHit("a", 0.9)],
            tolerance=1e-6,
        )


def test_rankings_reject_chunk_or_score_drift() -> None:
    local = [RankedHit("a", 0.9), RankedHit("b", 0.8)]
    with pytest.raises(AssertionError, match="chunk_id"):
        compare_rankings(local, [RankedHit("a", 0.9), RankedHit("x", 0.8)], 1e-6)
    with pytest.raises(AssertionError, match="score"):
        compare_rankings(
            local, [RankedHit("a", 0.9), RankedHit("b", 0.79)], 1e-6
        )


def test_regression_thresholds_fail_closed() -> None:
    thresholds = RegressionThresholds(0.5, 0.4, 0)
    assert_regression_thresholds(
        {"recall_at_5": 0.5, "mrr_at_10": 0.4, "failed_queries": 0},
        thresholds,
    )
    with pytest.raises(AssertionError, match="recall_at_5"):
        assert_regression_thresholds(
            {"recall_at_5": 0.49, "mrr_at_10": 0.9, "failed_queries": 0},
            thresholds,
        )


@pytest.mark.parametrize("field", ["recall_at_5", "mrr_at_10", "failed_queries"])
def test_regression_thresholds_require_all_metrics(field: str) -> None:
    metrics = {"recall_at_5": 0.5, "mrr_at_10": 0.4, "failed_queries": 0}
    del metrics[field]
    with pytest.raises(ValueError, match="fields"):
        assert_regression_thresholds(metrics, RegressionThresholds(0.5, 0.4, 0))


def test_repository_mock_regression_fixture_passes_frozen_thresholds() -> None:
    root = Path(__file__).parents[2]
    metrics = evaluate_fixture(root / "data/eval/s11-regression-v1.json")
    assert metrics == {"recall_at_5": 0.75, "mrr_at_10": 0.625, "failed_queries": 0}
    assert_regression_thresholds(metrics, RegressionThresholds(0.5, 0.4, 0))
