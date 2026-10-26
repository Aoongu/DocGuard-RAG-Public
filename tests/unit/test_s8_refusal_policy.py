from __future__ import annotations

from dataclasses import replace

from docguard.refusal.models import Action, ReasonCode, RefusalFeatures
from docguard.refusal.policy import RefusalThresholds, decide
from docguard.refusal.versioning import VersionRequest

THRESHOLDS = RefusalThresholds(top1=0.0, top3_mean=-1.0, coverage=0.5, margin=1.0)
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


def test_ordered_version_rules_precede_evidence_thresholds() -> None:
    unavailable = VersionRequest("9.0", "unavailable", ("9.0",), False)
    decision = decide(BASE, unavailable, THRESHOLDS)
    assert (decision.action, decision.reason_code) == (
        Action.REFUSE,
        ReasonCode.VERSION_NOT_AVAILABLE,
    )
    missing = VersionRequest(None, "none", (), False)
    conflict = decide(replace(BASE, paired_version_conflict=True), missing, THRESHOLDS)
    assert (conflict.action, conflict.reason_code) == (
        Action.CLARIFY,
        ReasonCode.VERSION_AMBIGUOUS,
    )


def test_explicit_comparison_can_answer_with_two_versions() -> None:
    request = VersionRequest(None, "ambiguous", ("17.11", "18.6"), True)
    features = replace(BASE, top_evidence_versions=("17.11", "18.6"), paired_version_conflict=True)
    assert decide(features, request, THRESHOLDS).action is Action.ANSWER


def test_insufficient_evidence_reason_distinguishes_domain_signal() -> None:
    request = VersionRequest(None, "none", (), False)
    weak = replace(
        BASE,
        top1_rerank_score=-3.0,
        top1_top2_margin=0.1,
        top3_mean_rerank_score=-4.0,
        evidence_term_coverage=0.2,
    )
    assert decide(weak, request, THRESHOLDS).reason_code is ReasonCode.NO_RELEVANT_EVIDENCE
    assert (
        decide(replace(weak, postgresql_domain_signal=False), request, THRESHOLDS).reason_code
        is ReasonCode.OUT_OF_DOMAIN
    )


def test_ablation_and_upstream_error_are_explicit() -> None:
    request = VersionRequest(None, "none", (), False)
    disabled = decide(BASE, request, THRESHOLDS, refusal_enabled=False)
    assert disabled.action is Action.ANSWER
    assert disabled.triggered_rule == "refusal_disabled_ablation"
    failed = decide(BASE, request, THRESHOLDS, upstream_error="timeout")
    assert (failed.action, failed.reason_code) == (Action.ERROR, ReasonCode.UPSTREAM_ERROR)
