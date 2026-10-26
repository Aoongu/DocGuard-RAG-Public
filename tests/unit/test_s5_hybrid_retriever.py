from __future__ import annotations

from collections.abc import Callable

import pytest

from docguard.retrieval.bm25_index import BM25SearchHit
from docguard.retrieval.hybrid import HybridRetriever
from docguard.retrieval.service import DenseSearchHit


def _dense(chunk_id: str = "dense", rank: int = 1) -> DenseSearchHit:
    return DenseSearchHit(
        rank=rank,
        chunk_id=chunk_id,
        score=0.8,
        postgresql_version="17.11",
        topic="installation_configuration",
        section_path=("Section",),
        canonical_url=f"https://www.postgresql.org/docs/17/{chunk_id}.html",
        anchor="main",
        text=f"text {chunk_id}",
    )


def _bm25(chunk_id: str = "bm25", rank: int = 1) -> BM25SearchHit:
    dense = _dense(chunk_id, rank)
    return BM25SearchHit(
        rank=rank,
        chunk_id=chunk_id,
        score=2.5,
        postgresql_version=dense.postgresql_version,
        topic=dense.topic,
        section_path=dense.section_path,
        canonical_url=dense.canonical_url,
        anchor=dense.anchor,
        text=dense.text,
    )


class FakeRoute:
    def __init__(self, hits: list[object]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, str | None, int, bool]] = []
        self.closed = False

    async def search(self, query: str, version: str | None, top_k: int, **switch) -> list[object]:
        self.calls.append((query, version, top_k, next(iter(switch.values()))))
        return self.hits[:top_k]

    async def close(self) -> None:
        self.closed = True


def _factory(route: FakeRoute, calls: list[str], name: str) -> Callable[[], FakeRoute]:
    def create() -> FakeRoute:
        calls.append(name)
        return route

    return create


@pytest.mark.parametrize("method", ["b0", "b1", "m1"])
async def test_methods_construct_and_call_only_required_routes(method: str) -> None:
    dense = FakeRoute([_dense("shared")])
    bm25 = FakeRoute([_bm25("shared")])
    factory_calls: list[str] = []
    retriever = HybridRetriever(
        _factory(dense, factory_calls, "dense"),
        _factory(bm25, factory_calls, "bm25"),
    )
    try:
        hits = await retriever.search(
            "wal_level",
            "17.11",
            5,
            method=method,
            dense_depth=20,
            bm25_depth=50,
            rrf_k=60,
        )
    finally:
        await retriever.close()

    expected = {"b0": ["bm25"], "b1": ["dense"], "m1": ["dense", "bm25"]}[method]
    assert factory_calls == expected
    assert [call[2] for call in dense.calls] == ([] if method == "b0" else [20 if method == "m1" else 5])
    assert [call[2] for call in bm25.calls] == ([] if method == "b1" else [50 if method == "m1" else 5])
    assert hits[0].sources == ({"b0": ("bm25",), "b1": ("dense",), "m1": ("dense", "bm25")}[method])


async def test_direct_methods_preserve_raw_score_without_rrf() -> None:
    dense = FakeRoute([_dense()])
    bm25 = FakeRoute([_bm25()])
    retriever = HybridRetriever(lambda: dense, lambda: bm25)
    try:
        b0 = await retriever.search("q", None, 1, method="b0")
        b1 = await retriever.search("q", None, 1, method="b1")
    finally:
        await retriever.close()

    assert b0[0].score == b0[0].bm25_score == 2.5 and b0[0].rrf_score is None
    assert b1[0].score == b1[0].dense_score == 0.8 and b1[0].rrf_score is None


async def test_invalid_method_fails_before_factories_are_called() -> None:
    calls: list[str] = []
    retriever = HybridRetriever(
        _factory(FakeRoute([]), calls, "dense"),
        _factory(FakeRoute([]), calls, "bm25"),
    )

    with pytest.raises(ValueError, match="method"):
        await retriever.search("query", None, 5, method="test")
    assert calls == []
