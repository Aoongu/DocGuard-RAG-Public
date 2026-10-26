from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from docguard.refusal.config import load_refusal_config
from docguard.refusal.models import RankedEvidence
from docguard.refusal.versioning import detect_version_conflict, parse_version_request

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_refusal_config(ROOT / "configs" / "refusal" / "s8-refusal-v1.json")


def _hit(rank: int, version: str, score: float, text: str) -> RankedEvidence:
    return RankedEvidence(
        rank=rank,
        chunk_id=f"c-{rank}-{version}",
        rerank_score=score,
        postgresql_version=version,
        canonical_url=f"https://www.postgresql.org/docs/{version.split('.')[0]}/sql-vacuum.html",
        anchor="SQL-VACUUM",
        text=text,
    )


def test_structured_version_precedes_question_text() -> None:
    request = parse_version_request("According to PostgreSQL 9.0", "17.11", CONFIG)
    assert request.requested_version == "17.11"
    assert request.status == "available"


def test_unavailable_and_explicit_comparison_requests() -> None:
    unsupported = parse_version_request("According to PostgreSQL 9.0, what changed?", None, CONFIG)
    assert unsupported.requested_version == "9.0"
    assert unsupported.status == "unavailable"
    comparison = parse_version_request("Compare PostgreSQL 17.11 and 18.6 VACUUM", None, CONFIG)
    assert comparison.comparison_requested is True
    assert comparison.status == "ambiguous"
    assert comparison.mentioned_versions == ("17.11", "18.6")


def test_conflict_requires_two_high_quality_same_source_differences() -> None:
    hits = [
        _hit(1, "17.11", 2.0, "VACUUM accepts FULL and FREEZE."),
        _hit(2, "18.6", 1.5, "VACUUM accepts FULL, FREEZE, ONLY and SKIP_LOCKED."),
    ]
    conflict = detect_version_conflict(hits, CONFIG)
    assert conflict.detected is True
    assert conflict.source_key.endswith("/sql-vacuum.html#SQL-VACUUM")
    assert conflict.token_jaccard < CONFIG.version_conflict.max_token_jaccard


def test_mixed_versions_or_low_quality_alone_do_not_imply_conflict() -> None:
    different_sources = [
        _hit(1, "17.11", 2.0, "same text"),
        replace(
            _hit(2, "18.6", 2.0, "different text"),
            canonical_url="https://www.postgresql.org/docs/18/sql-select.html",
        ),
    ]
    assert detect_version_conflict(different_sources, CONFIG).detected is False
    low = [_hit(1, "17.11", 2.0, "old syntax"), _hit(2, "18.6", -4.0, "new syntax")]
    assert detect_version_conflict(low, CONFIG).detected is False
