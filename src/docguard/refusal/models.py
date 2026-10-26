"""Immutable S8 decision records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    ANSWER = "answer"
    REFUSE = "refuse"
    CLARIFY = "clarify"
    ERROR = "error"


class ReasonCode(str, Enum):
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
    VERSION_NOT_AVAILABLE = "VERSION_NOT_AVAILABLE"
    VERSION_AMBIGUOUS = "VERSION_AMBIGUOUS"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    CITATION_VALIDATION_FAILED = "CITATION_VALIDATION_FAILED"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"


@dataclass(frozen=True, slots=True)
class RankedEvidence:
    rank: int
    chunk_id: str
    rerank_score: float
    postgresql_version: str
    canonical_url: str
    anchor: str
    text: str


@dataclass(frozen=True, slots=True)
class RefusalFeatures:
    top1_rerank_score: float
    top1_top2_margin: float
    top3_mean_rerank_score: float
    evidence_term_coverage: float
    requested_version: str | None
    requested_version_status: str
    top_evidence_versions: tuple[str, ...]
    best_score_by_version: tuple[tuple[str, float], ...]
    paired_version_conflict: bool
    postgresql_domain_signal: bool


@dataclass(frozen=True, slots=True)
class AnswerDecision:
    action: Action
    reason_code: ReasonCode | None
    confidence: float
    triggered_rule: str
    features: RefusalFeatures

