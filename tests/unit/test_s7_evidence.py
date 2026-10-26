from dataclasses import replace

import pytest

from docguard.generation.evidence import EvidenceMappingError, build_evidence
from docguard.retrieval.reranker import RerankedSearchHit
from docguard.retrieval.service import DenseSearchHit

CHUNK_A = "chk_" + "a" * 64
CHUNK_B = "chk_" + "b" * 64


def _dense(
    *,
    rank: int = 1,
    chunk_id: str = CHUNK_A,
    score: float = 0.9,
    text: str = "wal_level controls how much information is written to WAL.",
) -> DenseSearchHit:
    return DenseSearchHit(
        rank=rank,
        chunk_id=chunk_id,
        score=score,
        postgresql_version="17.11",
        topic="runtime-config",
        section_path=("Server Configuration", "Write Ahead Log"),
        canonical_url="https://www.postgresql.org/docs/17/runtime-config-wal.html",
        anchor="RUNTIME-CONFIG-WAL-SETTINGS",
        text=text,
    )


def _reranked() -> RerankedSearchHit:
    return RerankedSearchHit(
        rank=1,
        chunk_id=CHUNK_A,
        score=2.5,
        rerank_score=2.5,
        rerank_truncated=False,
        original_rrf_rank=3,
        rrf_score=0.08,
        sources=("dense", "bm25"),
        dense_rank=2,
        dense_score=0.81,
        bm25_rank=4,
        bm25_score=7.2,
        postgresql_version="18.6",
        topic="runtime-config",
        section_path=("Server Configuration", "Write Ahead Log"),
        canonical_url="https://www.postgresql.org/docs/18/runtime-config-wal.html",
        anchor="RUNTIME-CONFIG-WAL-SETTINGS",
        text="The amount of information written to WAL is controlled by wal_level.",
    )


def test_build_evidence_assigns_ids_and_preserves_dense_metadata() -> None:
    hits = [_dense(), _dense(rank=2, chunk_id=CHUNK_B, score=0.7)]

    evidence = build_evidence(hits, top_k=10, per_item_chars=4_000, total_chars=30_000)

    assert [item.evidence_id for item in evidence] == ["E01", "E02"]
    assert evidence[0].chunk_id == CHUNK_A
    assert evidence[0].rank == 1
    assert evidence[0].score == 0.9
    assert evidence[0].postgresql_version == "17.11"
    assert evidence[0].section_path == ("Server Configuration", "Write Ahead Log")
    assert evidence[0].canonical_url.endswith("runtime-config-wal.html")
    assert evidence[0].anchor == "RUNTIME-CONFIG-WAL-SETTINGS"
    assert evidence[0].original_text_chars == len(hits[0].text)
    assert evidence[0].text_truncated is False
    assert evidence[0].provenance.sources == ("dense",)
    assert evidence[0].provenance.dense_rank == 1
    assert evidence[0].provenance.dense_score == 0.9
    assert evidence[0].provenance.rerank_score is None


def test_build_evidence_preserves_rerank_provenance() -> None:
    evidence = build_evidence(
        [_reranked()], top_k=10, per_item_chars=4_000, total_chars=30_000
    )

    provenance = evidence[0].provenance
    assert provenance.sources == ("dense", "bm25")
    assert provenance.dense_rank == 2
    assert provenance.dense_score == 0.81
    assert provenance.bm25_rank == 4
    assert provenance.bm25_score == 7.2
    assert provenance.rrf_score == 0.08
    assert provenance.rerank_score == 2.5
    assert provenance.rerank_truncated is False
    assert provenance.original_rrf_rank == 3


def test_build_evidence_applies_fair_deterministic_character_budgets() -> None:
    hits = [
        _dense(rank=1, chunk_id=CHUNK_A, text="a" * 100),
        _dense(rank=2, chunk_id=CHUNK_B, text="b" * 100),
    ]

    evidence = build_evidence(hits, top_k=2, per_item_chars=80, total_chars=100)

    assert [len(item.text) for item in evidence] == [50, 50]
    assert [item.original_text_chars for item in evidence] == [100, 100]
    assert all(item.text_truncated for item in evidence)
    assert "".join(item.text for item in evidence) == "a" * 50 + "b" * 50
    assert build_evidence(hits, top_k=2, per_item_chars=80, total_chars=100) == evidence


def test_build_evidence_honors_top_k_without_reordering() -> None:
    hits = [_dense(), _dense(rank=2, chunk_id=CHUNK_B)]
    assert len(build_evidence(hits, top_k=1, per_item_chars=80, total_chars=100)) == 1


@pytest.mark.parametrize(
    ("hits", "message"),
    [
        ([_dense(), _dense(rank=2, chunk_id=CHUNK_A)], "duplicate chunk_id"),
        ([_dense(), _dense(rank=1, chunk_id=CHUNK_B)], "duplicate rank"),
        ([_dense(rank=0)], "rank"),
        ([_dense(score=float("nan"))], "finite"),
        ([_dense(chunk_id="bad")], "chunk_id"),
        ([_dense(text="  ")], "text"),
        ([replace(_dense(), canonical_url="http://example.com")], "canonical_url"),
        ([replace(_dense(), section_path=())], "section_path"),
    ],
)
def test_build_evidence_rejects_invalid_hits(
    hits: list[DenseSearchHit], message: str
) -> None:
    with pytest.raises(EvidenceMappingError, match=message):
        build_evidence(hits, top_k=10, per_item_chars=4_000, total_chars=30_000)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"top_k": 0, "per_item_chars": 10, "total_chars": 10}, "top_k"),
        ({"top_k": 1, "per_item_chars": 0, "total_chars": 10}, "per_item_chars"),
        ({"top_k": 2, "per_item_chars": 10, "total_chars": 1}, "total_chars"),
    ],
)
def test_build_evidence_rejects_invalid_budgets(
    kwargs: dict[str, int], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        build_evidence([_dense()], **kwargs)
