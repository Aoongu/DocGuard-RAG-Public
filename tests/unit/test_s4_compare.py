import pytest

from docguard.evaluation.compare import paired_bootstrap


def test_paired_bootstrap_is_reproducible_and_reports_95_percent_ci() -> None:
    baseline = {f"q{i}": float(i % 2) for i in range(20)}
    candidate = {f"q{i}": min(1.0, value + 0.25) for i, value in enumerate(baseline.values())}

    first = paired_bootstrap(
        baseline,
        candidate,
        seed=20261014,
        samples=2000,
        confidence_level=0.95,
    )
    second = paired_bootstrap(
        baseline,
        candidate,
        seed=20261014,
        samples=2000,
        confidence_level=0.95,
    )

    assert first == second
    assert first["paired_question_count"] == 20
    assert first["confidence_level"] == 0.95
    assert first["mean_difference"] == pytest.approx(0.125)
    assert first["ci_lower"] <= first["mean_difference"] <= first["ci_upper"]


def test_paired_bootstrap_requires_same_nonempty_question_ids() -> None:
    with pytest.raises(ValueError, match="same question IDs"):
        paired_bootstrap({"q1": 1.0}, {"q2": 1.0}, 1, 1000, 0.95)
    with pytest.raises(ValueError, match="must not be empty"):
        paired_bootstrap({}, {}, 1, 1000, 0.95)
