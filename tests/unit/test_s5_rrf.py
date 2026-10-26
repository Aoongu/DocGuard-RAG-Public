from __future__ import annotations

from dataclasses import replace

import pytest

from docguard.retrieval.bm25_index import BM25SearchHit
from docguard.retrieval.hybrid import HybridRetrievalError, reciprocal_rank_fusion
from docguard.retrieval.service import DenseSearchHit


def _dense(chunk_id: str, rank: int, score: float = 0.9) -> DenseSearchHit:
    return DenseSearchHit(
        rank=rank,
        chunk_id=chunk_id,
        score=score,
        postgresql_version="17.11",
        topic="installation_configuration",
        section_path=("Section",),
        canonical_url=f"https://www.postgresql.org/docs/17/{chunk_id}.html",
        anchor="main",
        text=f"text {chunk_id}",
    )


def _bm25(chunk_id: str, rank: int, score: float = 2.0) -> BM25SearchHit:
    dense = _dense(chunk_id, rank)
    return BM25SearchHit(
        rank=rank,
        chunk_id=chunk_id,
        score=score,
        postgresql_version=dense.postgresql_version,
        topic=dense.topic,
        section_path=dense.section_path,
        canonical_url=dense.canonical_url,
        anchor=dense.anchor,
        text=dense.text,
    )


def test_rrf_fuses_overlap_and_preserves_route_provenance() -> None:
    hits = reciprocal_rank_fusion(
        [_dense("a", 1), _dense("b", 2)],
        [_bm25("b", 1), _bm25("c", 2)],
        rrf_k=20,
        top_k=3,
    )

    assert [hit.chunk_id for hit in hits] == ["b", "a", "c"]
    overlap = hits[0]
    assert overlap.rrf_score == pytest.approx(1 / 22 + 1 / 21)
    assert overlap.score == overlap.rrf_score
    assert overlap.sources == ("dense", "bm25")
    assert overlap.dense_rank == 2 and overlap.bm25_rank == 1
    assert overlap.dense_score == 0.9 and overlap.bm25_score == 2.0


@pytest.mark.parametrize("route", ["dense", "bm25"])
def test_rrf_accepts_one_route_and_uses_chunk_id_tie_break(route: str) -> None:
    dense = [_dense("b", 1), _dense("a", 1)] if route == "dense" else []
    bm25 = [_bm25("b", 1), _bm25("a", 1)] if route == "bm25" else []

    hits = reciprocal_rank_fusion(dense, bm25, rrf_k=60, top_k=2)

    assert [hit.chunk_id for hit in hits] == ["a", "b"]
    assert all(hit.sources == (route,) for hit in hits)


def test_rrf_rejects_duplicate_chunks_and_inconsistent_metadata() -> None:
    with pytest.raises(HybridRetrievalError, match="duplicate"):
        reciprocal_rank_fusion([_dense("a", 1), _dense("a", 2)], [], rrf_k=60, top_k=2)
    mismatched = replace(_bm25("a", 1), text="different")
    with pytest.raises(HybridRetrievalError, match="metadata"):
        reciprocal_rank_fusion([_dense("a", 1)], [mismatched], rrf_k=60, top_k=2)


@pytest.mark.parametrize(("rrf_k", "top_k"), [(0, 1), (60, 0), (True, 1)])
def test_rrf_rejects_invalid_parameters(rrf_k: int, top_k: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        reciprocal_rank_fusion([], [], rrf_k=rrf_k, top_k=top_k)
