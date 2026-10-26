"""Pure, label-free feature extraction for S8."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from docguard.refusal.config import RefusalConfig
from docguard.refusal.models import RankedEvidence, RefusalFeatures


class FeatureExtractionError(ValueError):
    """Raised when upstream evidence violates the S8 input contract."""


def _normalize_token(token: str) -> str:
    value = token.lower()
    if len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def _tokens(text: str, config: RefusalConfig) -> set[str]:
    stopwords = set(config.stopwords)
    return {
        _normalize_token(match.group(0))
        for match in re.finditer(config.token_pattern, text.lower())
        if match.group(0) not in stopwords
    }


def _validate_hits(hits: Sequence[RankedEvidence], config: RefusalConfig) -> list[RankedEvidence]:
    if not hits:
        raise FeatureExtractionError("at least one ranked hit is required")
    ordered = sorted(hits, key=lambda item: item.rank)
    if [item.rank for item in ordered] != list(range(1, len(ordered) + 1)):
        raise FeatureExtractionError("hit ranks must be unique and contiguous from one")
    chunk_ids: set[str] = set()
    for hit in ordered:
        if not hit.chunk_id or hit.chunk_id in chunk_ids:
            raise FeatureExtractionError("hit chunk IDs must be non-empty and unique")
        chunk_ids.add(hit.chunk_id)
        if not math.isfinite(hit.rerank_score):
            raise FeatureExtractionError("rerank scores must be finite")
        if hit.postgresql_version not in config.supported_versions:
            raise FeatureExtractionError("hit version is outside the frozen corpus")
        if not hit.text.strip() or not hit.canonical_url or not hit.anchor:
            raise FeatureExtractionError("hit provenance and text must be non-empty")
    return ordered


def extract_refusal_features(
    question: str,
    hits: Sequence[RankedEvidence],
    config: RefusalConfig,
    *,
    requested_version: str | None,
    requested_version_status: str,
    paired_version_conflict: bool,
) -> RefusalFeatures:
    """Extract the frozen S8 feature vector without consulting evaluation labels."""
    if not isinstance(question, str) or not question.strip() or "\0" in question:
        raise FeatureExtractionError("question must be a non-empty string without NUL")
    if requested_version_status not in {"none", "available", "unavailable", "ambiguous"}:
        raise FeatureExtractionError("unknown requested version status")
    ordered = _validate_hits(hits, config)
    top = ordered[:3]
    question_terms = _tokens(question, config)
    evidence_terms = _tokens(" ".join(item.text for item in top), config)
    coverage = (
        len(question_terms & evidence_terms) / len(question_terms) if question_terms else 0.0
    )
    score_by_version: dict[str, float] = {}
    for hit in ordered:
        score_by_version[hit.postgresql_version] = max(
            hit.rerank_score,
            score_by_version.get(hit.postgresql_version, -math.inf),
        )
    raw_question_tokens = {
        _normalize_token(match.group(0))
        for match in re.finditer(config.token_pattern, question.lower())
    }
    normalized_domain_terms = {_normalize_token(item) for item in config.domain_terms}
    return RefusalFeatures(
        top1_rerank_score=ordered[0].rerank_score,
        top1_top2_margin=(
            ordered[0].rerank_score - ordered[1].rerank_score
            if len(ordered) > 1
            else 0.0
        ),
        top3_mean_rerank_score=sum(item.rerank_score for item in top) / len(top),
        evidence_term_coverage=coverage,
        requested_version=requested_version,
        requested_version_status=requested_version_status,
        top_evidence_versions=tuple(sorted(score_by_version)),
        best_score_by_version=tuple(sorted(score_by_version.items())),
        paired_version_conflict=paired_version_conflict,
        postgresql_domain_signal=bool(raw_question_tokens & normalized_domain_terms),
    )
