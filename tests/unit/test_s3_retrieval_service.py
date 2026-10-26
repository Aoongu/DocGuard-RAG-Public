from __future__ import annotations

import math
from uuid import uuid4

import pytest
from ragbits.core.vector_stores.base import VectorStoreEntry, VectorStoreResult


class FakeStore:
    def __init__(self, results: list[VectorStoreResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    async def retrieve(self, text: str, options: object) -> list[VectorStoreResult]:
        self.calls.append((text, options))
        return self.results

    async def close(self) -> None:
        self.closed = True


def _result(chunk_id: str, version: str, score: float) -> VectorStoreResult:
    return VectorStoreResult(
        entry=VectorStoreEntry(
            id=uuid4(),
            text=f"text for {chunk_id}",
            metadata={
                "chunk_id": chunk_id,
                "postgresql_version": version,
                "topic": "sql",
                "section_path": ["SQL", "SELECT"],
                "canonical_url": f"https://example.test/{version}/sql-select.html",
                "anchor": "SQL-SELECT",
                "ordinal": 0,
                "content_sha256": "a" * 64,
            },
        ),
        vector=[],
        score=score,
    )


@pytest.mark.asyncio
async def test_dense_switch_returns_without_constructing_store() -> None:
    from docguard.retrieval.service import DenseRetriever

    calls = 0

    def factory() -> FakeStore:
        nonlocal calls
        calls += 1
        return FakeStore([])

    retriever = DenseRetriever(factory)
    assert await retriever.search("SELECT syntax", None, 5, dense_enabled=False) == []
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "version", "top_k"),
    [("", None, 5), ("   ", None, 5), ("ok", "16", 5), ("ok", None, 0)],
)
async def test_search_rejects_invalid_request(
    query: str, version: str | None, top_k: int
) -> None:
    from docguard.retrieval.service import DenseRetriever

    retriever = DenseRetriever(lambda: FakeStore([]))
    with pytest.raises(ValueError):
        await retriever.search(query, version, top_k, dense_enabled=True)


@pytest.mark.asyncio
async def test_search_maps_and_stably_orders_results() -> None:
    from docguard.retrieval.service import DenseRetriever

    store = FakeStore(
        [
            _result("chk_" + "b" * 64, "17.11", 0.75),
            _result("chk_" + "a" * 64, "17.11", 0.75),
            _result("chk_" + "c" * 64, "17.11", 0.5),
        ]
    )
    retriever = DenseRetriever(lambda: store)
    hits = await retriever.search("SELECT syntax", "17.11", 3)

    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert [hit.chunk_id for hit in hits] == [
        "chk_" + "a" * 64,
        "chk_" + "b" * 64,
        "chk_" + "c" * 64,
    ]
    assert all(hit.postgresql_version == "17.11" for hit in hits)
    assert all(math.isfinite(hit.score) for hit in hits)
    assert hits[0].section_path == ("SQL", "SELECT")
    options = store.calls[0][1]
    assert options.k == 3  # type: ignore[attr-defined]
    assert options.where == {"postgresql_version": "17.11"}  # type: ignore[attr-defined]
    await retriever.close()
    assert store.closed


@pytest.mark.asyncio
async def test_search_rejects_version_leakage_and_nonfinite_score() -> None:
    from docguard.retrieval.service import DenseRetrievalError, DenseRetriever

    leaking = DenseRetriever(
        lambda: FakeStore([_result("chk_" + "d" * 64, "18.6", 0.8)])
    )
    with pytest.raises(DenseRetrievalError, match="version leakage"):
        await leaking.search("SELECT syntax", "17.11", 5)

    nonfinite = DenseRetriever(
        lambda: FakeStore([_result("chk_" + "e" * 64, "17.11", float("nan"))])
    )
    with pytest.raises(DenseRetrievalError, match="finite"):
        await nonfinite.search("SELECT syntax", None, 5)
