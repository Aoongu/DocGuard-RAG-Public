"""Version-aware refusal decisions for DocGuard-RAG."""

from docguard.refusal.config import RefusalConfig, load_refusal_config
from docguard.refusal.models import (
    Action,
    AnswerDecision,
    RankedEvidence,
    ReasonCode,
    RefusalFeatures,
)
from docguard.refusal.pipeline import M4Pipeline, M4Result
from docguard.refusal.policy import RefusalThresholds, decide
from docguard.refusal.versioning import (
    VersionConflict,
    VersionRequest,
    detect_version_conflict,
    parse_version_request,
)

__all__ = [
    "Action",
    "AnswerDecision",
    "M4Pipeline",
    "M4Result",
    "RankedEvidence",
    "ReasonCode",
    "RefusalConfig",
    "RefusalFeatures",
    "RefusalThresholds",
    "VersionConflict",
    "VersionRequest",
    "decide",
    "detect_version_conflict",
    "load_refusal_config",
    "parse_version_request",
]
