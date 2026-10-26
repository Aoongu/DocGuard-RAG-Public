"""Ordered transparent S8 refusal policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from docguard.refusal.models import Action, AnswerDecision, ReasonCode, RefusalFeatures
from docguard.refusal.versioning import VersionRequest


@dataclass(frozen=True, slots=True, order=True)
class RefusalThresholds:
    top1: float
    top3_mean: float
    coverage: float
    margin: float

    def __post_init__(self) -> None:
        values = (self.top1, self.top3_mean, self.coverage, self.margin)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("refusal thresholds must be finite")
        if not 0.0 <= self.coverage <= 1.0 or self.margin < 0.0:
            raise ValueError("coverage and margin thresholds are invalid")


def _decision(
    action: Action,
    reason: ReasonCode | None,
    confidence: float,
    rule: str,
    features: RefusalFeatures,
) -> AnswerDecision:
    return AnswerDecision(action, reason, min(1.0, max(0.0, confidence)), rule, features)


def _evidence_confidence(features: RefusalFeatures, thresholds: RefusalThresholds) -> float:
    score_margin = max(
        features.top1_rerank_score - thresholds.top1,
        min(
            features.top3_mean_rerank_score - thresholds.top3_mean,
            features.evidence_term_coverage - thresholds.coverage,
        ),
        min(
            features.top1_top2_margin - thresholds.margin,
            features.evidence_term_coverage - thresholds.coverage,
        ),
    )
    return 0.5 + 0.25 * math.tanh(score_margin)


def decide(
    features: RefusalFeatures,
    version_request: VersionRequest,
    thresholds: RefusalThresholds,
    *,
    refusal_enabled: bool = True,
    version_filter_enabled: bool = True,
    upstream_error: str | None = None,
) -> AnswerDecision:
    """Apply version hard rules before the calibrated evidence rule."""
    if upstream_error is not None:
        return _decision(Action.ERROR, ReasonCode.UPSTREAM_ERROR, 1.0, "upstream_error", features)
    if version_request.status == "unavailable":
        return _decision(
            Action.REFUSE,
            ReasonCode.VERSION_NOT_AVAILABLE,
            1.0,
            "requested_version_unavailable",
            features,
        )
    if version_request.comparison_requested:
        comparison_versions = set(version_request.mentioned_versions)
        if comparison_versions.issubset(features.top_evidence_versions):
            return _decision(Action.ANSWER, None, 1.0, "explicit_version_comparison", features)
    if version_request.status == "ambiguous":
        return _decision(
            Action.CLARIFY,
            ReasonCode.VERSION_AMBIGUOUS,
            1.0,
            "multiple_versions_without_comparison",
            features,
        )
    if version_request.requested_version is None and features.paired_version_conflict:
        return _decision(
            Action.CLARIFY,
            ReasonCode.VERSION_AMBIGUOUS,
            1.0,
            "missing_version_with_conflicting_evidence",
            features,
        )
    if (
        version_filter_enabled
        and version_request.status == "available"
        and set(features.top_evidence_versions) != {version_request.requested_version}
    ):
        return _decision(
            Action.CLARIFY,
            ReasonCode.VERSION_CONFLICT,
            1.0,
            "requested_version_evidence_conflict",
            features,
        )
    if not refusal_enabled:
        return _decision(Action.ANSWER, None, 0.0, "refusal_disabled_ablation", features)

    sufficient = (
        features.top1_rerank_score >= thresholds.top1
        or (
            features.top3_mean_rerank_score >= thresholds.top3_mean
            and features.evidence_term_coverage >= thresholds.coverage
        )
        or (
            features.top1_top2_margin >= thresholds.margin
            and features.evidence_term_coverage >= thresholds.coverage
        )
    )
    evidence_confidence = _evidence_confidence(features, thresholds)
    if sufficient:
        return _decision(Action.ANSWER, None, evidence_confidence, "evidence_sufficient", features)
    if not features.postgresql_domain_signal:
        reason = ReasonCode.OUT_OF_DOMAIN
        rule = "evidence_insufficient_without_domain_signal"
    elif (
        features.top1_rerank_score < thresholds.top1
        and features.top3_mean_rerank_score < thresholds.top3_mean
    ):
        reason = ReasonCode.NO_RELEVANT_EVIDENCE
        rule = "reranker_evidence_below_threshold"
    else:
        reason = ReasonCode.EVIDENCE_INSUFFICIENT
        rule = "combined_evidence_below_threshold"
    return _decision(Action.REFUSE, reason, 1.0 - evidence_confidence, rule, features)
