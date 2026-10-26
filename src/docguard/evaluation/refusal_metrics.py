"""Pure metrics for S8 answer/refuse/clarify/error decisions."""

from __future__ import annotations

from dataclasses import dataclass

from docguard.refusal.models import Action

_ACTIONS = tuple(Action)
_REFUSAL_ACTIONS = {Action.REFUSE, Action.CLARIFY}


@dataclass(frozen=True, slots=True)
class MetricRecord:
    question_id: str
    expected: Action
    predicted: Action
    refusal_score: float


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    accepted: int
    coverage: float
    selective_risk: float
    max_refusal_score: float


@dataclass(frozen=True, slots=True)
class RefusalMetrics:
    n: int
    refusal_precision: float | None
    refusal_recall: float | None
    refusal_f1: float | None
    macro_f1: float
    false_accept_count: int
    false_reject_count: int
    answerable_coverage: float | None
    roc_auc: float | None
    confusion_matrix: dict[str, dict[str, int]]
    risk_coverage: tuple[RiskCoveragePoint, ...]


def expected_action(answerable: bool, reason_code: str | None) -> Action:
    if answerable:
        if reason_code is not None:
            raise ValueError("answerable examples cannot have a refusal reason")
        return Action.ANSWER
    if reason_code in {
        "NO_EVIDENCE",
        "NO_RELEVANT_EVIDENCE",
        "UNSUPPORTED_VERSION",
        "VERSION_NOT_AVAILABLE",
        "OUT_OF_DOMAIN",
        "EVIDENCE_INSUFFICIENT",
    }:
        return Action.REFUSE
    if reason_code in {"VERSION_REQUIRED", "VERSION_AMBIGUOUS", "VERSION_CONFLICT"}:
        return Action.CLARIFY
    raise ValueError(f"unknown boundary reason code: {reason_code}")


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _auc(records: list[MetricRecord]) -> float | None:
    positive = [row for row in records if row.expected is not Action.ANSWER]
    negative = [row for row in records if row.expected is Action.ANSWER]
    if not positive or not negative:
        return None
    wins = 0.0
    for left in positive:
        for right in negative:
            if left.refusal_score > right.refusal_score:
                wins += 1.0
            elif left.refusal_score == right.refusal_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def _risk_coverage(records: list[MetricRecord]) -> tuple[RiskCoveragePoint, ...]:
    ordered = sorted(records, key=lambda row: (row.refusal_score, row.question_id))
    boundary = 0
    points: list[RiskCoveragePoint] = []
    for index, row in enumerate(ordered, 1):
        boundary += row.expected is not Action.ANSWER
        points.append(
            RiskCoveragePoint(
                accepted=index,
                coverage=index / len(ordered),
                selective_risk=boundary / index,
                max_refusal_score=row.refusal_score,
            )
        )
    return tuple(points)


def aggregate_refusal_metrics(records: list[MetricRecord]) -> RefusalMetrics:
    if not records:
        raise ValueError("at least one metric record is required")
    if len({row.question_id for row in records}) != len(records):
        raise ValueError("metric question IDs must be unique")
    matrix = {
        expected.value: {predicted.value: 0 for predicted in _ACTIONS}
        for expected in _ACTIONS
    }
    for row in records:
        if not 0.0 <= row.refusal_score <= 1.0:
            raise ValueError("refusal_score must be in [0, 1]")
        matrix[row.expected.value][row.predicted.value] += 1

    true_positive = sum(
        row.expected is not Action.ANSWER and row.predicted in _REFUSAL_ACTIONS
        for row in records
    )
    predicted_positive = sum(row.predicted in _REFUSAL_ACTIONS for row in records)
    actual_positive = sum(row.expected is not Action.ANSWER for row in records)
    precision = _safe_ratio(true_positive, predicted_positive)
    recall = _safe_ratio(true_positive, actual_positive)

    class_f1: list[float] = []
    for action in _ACTIONS:
        true = matrix[action.value][action.value]
        predicted = sum(matrix[expected.value][action.value] for expected in _ACTIONS)
        actual = sum(matrix[action.value].values())
        value = _f1(_safe_ratio(true, predicted), _safe_ratio(true, actual))
        class_f1.append(0.0 if value is None else value)

    expected_answers = sum(row.expected is Action.ANSWER for row in records)
    correct_answers = sum(
        row.expected is Action.ANSWER and row.predicted is Action.ANSWER for row in records
    )
    return RefusalMetrics(
        n=len(records),
        refusal_precision=precision,
        refusal_recall=recall,
        refusal_f1=_f1(precision, recall),
        macro_f1=sum(class_f1) / len(class_f1),
        false_accept_count=sum(
            row.expected is not Action.ANSWER and row.predicted is Action.ANSWER
            for row in records
        ),
        false_reject_count=sum(
            row.expected is Action.ANSWER and row.predicted is not Action.ANSWER
            for row in records
        ),
        answerable_coverage=_safe_ratio(correct_answers, expected_answers),
        roc_auc=_auc(records),
        confusion_matrix=matrix,
        risk_coverage=_risk_coverage(records),
    )
