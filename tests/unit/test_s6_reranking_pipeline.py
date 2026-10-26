from __future__ import annotations

from pathlib import Path

import pytest

from docguard.retrieval.hybrid import HybridSearchHit
from docguard.retrieval.reranker import PairScore, RerankerError
from docguard.retrieval.reranker_config import load_reranker_config
from docguard.retrieval.reranking_pipeline import RerankingRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_reranker_config(
    PROJECT_ROOT / "configs" / "retrieval" / "s6-reranker-v1.json"
)


def _hit(rank: int) -> HybridSearchHit:
    return HybridSearchHit(
        rank=rank,
        chunk_id=f"chunk-{rank:02d}",
        score=1 / (20 + rank),
        rrf_score=1 / (20 + rank),
        sources=("dense", "bm25"),
        dense_rank=rank,
        dense_score=1 - rank / 100,
        bm25_rank=rank,
        bm25_score=2 - rank / 100,
        postgresql_version="18.6",
        topic="sql_commands",
        section_path=("SELECT",),
        canonical_url="https://www.postgresql.org/docs/18/sql-select.html",
        anchor=f"A-{rank}",
        text=f"passage {rank}",
    )


class FakeHybrid:
    def __init__(self, count: int = 50) -> None:
        self.hits = [_hit(index) for index in range(1, count + 1)]
        self.calls: list[tuple[str, str | None, int, dict[str, object]]] = []
        self.closed = False

    async def search(
        self, query: str, version: str | None, top_k: int, **options: object
    ) -> list[HybridSearchHit]:
        self.calls.append((query, version, top_k, options))
        return self.hits[:top_k]

    async def close(self) -> None:
        self.closed = True


class FakeReranker:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.closed = False
        self.fail = fail

    def score_pairs(self, query: str, passages: list[str]) -> list[PairScore]:
        self.calls.append((query, passages))
        if self.fail:
            raise RerankerError("controlled failure")
        return [PairScore(float(index), False) for index in range(len(passages))]

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_m2_uses_frozen_m1_parameters_and_reranks_requested_depth() -> None:
    hybrid = FakeHybrid()
    reranker = FakeReranker()
    factory_calls = 0

    def factory() -> FakeReranker:
        nonlocal factory_calls
        factory_calls += 1
        return reranker

    retrieval = RerankingRetriever(hybrid, factory, CONFIG)
    hits = await retrieval.search(
        "create index", "18.6", 10, method="m2", rerank_depth=20
    )

    assert hybrid.calls == [
        (
            "create index",
            "18.6",
            20,
            {"method": "m1", "dense_depth": 50, "bm25_depth": 20, "rrf_k": 20},
        )
    ]
    assert factory_calls == 1
    assert len(reranker.calls[0][1]) == 20
    assert len(hits) == 10
    assert [item.chunk_id for item in hits] == [
        f"chunk-{index:02d}" for index in range(20, 10, -1)
    ]
    assert all(item.original_rrf_rank >= 11 for item in hits)


@pytest.mark.asyncio
async def test_disabled_m2_is_exact_m1_ablation_without_model_construction() -> None:
    hybrid = FakeHybrid()

    def forbidden_factory() -> FakeReranker:
        raise AssertionError("disabled reranking constructed the model")

    retrieval = RerankingRetriever(hybrid, forbidden_factory, CONFIG)
    hits = await retrieval.search(
        "query", "17.11", 10, method="m2", rerank_depth=20, rerank_enabled=False
    )

    assert [item.chunk_id for item in hits] == [f"chunk-{index:02d}" for index in range(1, 11)]
    assert hybrid.calls[0][2] == 10
    assert hybrid.calls[0][3]["method"] == "m1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "depth", "top_k", "message"),
    [
        ("b1", 20, 10, "method"),
        ("m2", 30, 10, "rerank_depth"),
        ("m2", 10, 20, "top_k"),
    ],
)
async def test_m2_rejects_invalid_route_parameters(
    method: str, depth: int, top_k: int, message: str
) -> None:
    retrieval = RerankingRetriever(FakeHybrid(), FakeReranker, CONFIG)

    with pytest.raises(ValueError, match=message):
        await retrieval.search("query", None, top_k, method=method, rerank_depth=depth)


@pytest.mark.asyncio
async def test_m2_propagates_reranker_failure_and_closes_constructed_dependencies() -> None:
    hybrid = FakeHybrid()
    reranker = FakeReranker(fail=True)
    retrieval = RerankingRetriever(hybrid, lambda: reranker, CONFIG)

    with pytest.raises(RerankerError, match="controlled"):
        await retrieval.search("query", None, 10, method="m2", rerank_depth=10)

    await retrieval.close()
    assert hybrid.closed is True
    assert reranker.closed is True
