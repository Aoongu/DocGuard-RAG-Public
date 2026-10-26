from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from docguard.refusal.config import load_refusal_config
from docguard.refusal.features import FeatureExtractionError, extract_refusal_features
from docguard.refusal.models import RankedEvidence

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_refusal_config(ROOT / "configs" / "refusal" / "s8-refusal-v1.json")


def _hit(rank: int, score: float, *, version: str = "17.11", text: str = "") -> RankedEvidence:
    return RankedEvidence(
        rank=rank,
        chunk_id=f"chunk-{rank}-{version}",
        rerank_score=score,
        postgresql_version=version,
        canonical_url=f"https://www.postgresql.org/docs/{version.split('.')[0]}/sql-select.html",
        anchor="SQL-SELECT",
        text=text or "SELECT reads rows from a PostgreSQL table in a transaction.",
    )


def test_extracts_frozen_score_coverage_and_version_features() -> None:
    features = extract_refusal_features(
        "How does SELECT read rows from a table?",
        [_hit(1, 2.0), _hit(2, 1.25, version="18.6"), _hit(3, -1.0)],
        CONFIG,
        requested_version="17.11",
        requested_version_status="available",
        paired_version_conflict=False,
    )
    assert features.top1_rerank_score == 2.0
    assert features.top1_top2_margin == 0.75
    assert features.top3_mean_rerank_score == pytest.approx(0.75)
    assert features.evidence_term_coverage == 1.0
    assert features.top_evidence_versions == ("17.11", "18.6")
    assert features.best_score_by_version == (("17.11", 2.0), ("18.6", 1.25))
    assert features.postgresql_domain_signal is True


@pytest.mark.parametrize(
    "hits",
    [
        [],
        [_hit(1, math.nan)],
        [_hit(1, 1.0), replace(_hit(2, 0.5), rank=1)],
        [_hit(1, 1.0), replace(_hit(2, 0.5), chunk_id="chunk-1-17.11")],
        [_hit(1, 1.0, version="16.2")],
        [replace(_hit(1, 1.0), text="")],
    ],
)
def test_feature_extraction_fails_closed(hits: list[RankedEvidence]) -> None:
    with pytest.raises(FeatureExtractionError):
        extract_refusal_features(
            "question",
            hits,
            CONFIG,
            requested_version=None,
            requested_version_status="none",
            paired_version_conflict=False,
        )
