"""Unified B0/B1/M1 retrieval and deterministic reciprocal-rank fusion."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from docguard.retrieval.bm25_index import BM25SearchHit
from docguard.retrieval.service import DenseSearchHit


class HybridRetrievalError(RuntimeError):
    """Raised when route outputs cannot be fused safely."""


class SearchRoute(Protocol):
    async def search(
        self, query: str, version: str | None, top_k: int, **switch: bool
    ) -> list[object]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class HybridSearchHit:
    rank: int
    chunk_id: str
    score: float
    rrf_score: float | None
    sources: tuple[str, ...]
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    postgresql_version: str
    topic: str
    section_path: tuple[str, ...]
    canonical_url: str
    anchor: str
    text: str


RouteHit = TypeVar("RouteHit", DenseSearchHit, BM25SearchHit)


def _metadata(hit: DenseSearchHit | BM25SearchHit) -> tuple[object, ...]:
    return (
        hit.postgresql_version,
        hit.topic,
        hit.section_path,
        hit.canonical_url,
        hit.anchor,
        hit.text,
    )


def _route_map(hits: Sequence[RouteHit], route: str) -> dict[str, RouteHit]:
    mapped: dict[str, RouteHit] = {}
    for hit in hits:
        if type(hit.rank) is not int or hit.rank <= 0:
            raise HybridRetrievalError(f"{route} rank must be a positive integer")
        if not math.isfinite(hit.score):
            raise HybridRetrievalError(f"{route} score must be finite")
        if hit.chunk_id in mapped:
            raise HybridRetrievalError(f"duplicate {route} chunk_id: {hit.chunk_id}")
        mapped[hit.chunk_id] = hit
    return mapped


def rrf_scores(
    dense_ranks: dict[str, int], bm25_ranks: dict[str, int], *, rrf_k: int
) -> dict[str, float]:
    """Domain-independent RRF arithmetic shared by frozen and imported corpora."""
    if type(rrf_k) is not int or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    for ranks in (dense_ranks, bm25_ranks):
        if any(type(rank) is not int or rank <= 0 for rank in ranks.values()):
            raise ValueError("RRF ranks must be positive integers")
    return {
        key: sum(
            1 / (rrf_k + ranks[key])
            for ranks in (dense_ranks, bm25_ranks)
            if key in ranks
        )
        for key in set(dense_ranks) | set(bm25_ranks)
    }


def reciprocal_rank_fusion(
    dense_hits: Sequence[DenseSearchHit],
    bm25_hits: Sequence[BM25SearchHit],
    *,
    rrf_k: int,
    top_k: int,
) -> list[HybridSearchHit]:
    if type(rrf_k) is not int or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    dense = _route_map(dense_hits, "dense")
    bm25 = _route_map(bm25_hits, "bm25")
    shared = set(dense) & set(bm25)
    for chunk_id in shared:
        if _metadata(dense[chunk_id]) != _metadata(bm25[chunk_id]):
            raise HybridRetrievalError(
                f"route metadata differs for shared chunk: {chunk_id}"
            )

    scored: list[tuple[float, str]] = []
    for chunk_id, score in rrf_scores(
        {key: hit.rank for key, hit in dense.items()},
        {key: hit.rank for key, hit in bm25.items()},
        rrf_k=rrf_k,
    ).items():
        if not math.isfinite(score):
            raise HybridRetrievalError("RRF score must be finite")
        scored.append((score, chunk_id))
    scored.sort(key=lambda item: (-item[0], item[1]))

    output: list[HybridSearchHit] = []
    for rank, (score, chunk_id) in enumerate(scored[:top_k], start=1):
        dense_hit = dense.get(chunk_id)
        bm25_hit = bm25.get(chunk_id)
        source = dense_hit if dense_hit is not None else bm25_hit
        if source is None:
            raise HybridRetrievalError("RRF candidate has no source route")
        sources = tuple(
            name
            for name, hit in (("dense", dense_hit), ("bm25", bm25_hit))
            if hit is not None
        )
        output.append(
            HybridSearchHit(
                rank=rank,
                chunk_id=chunk_id,
                score=score,
                rrf_score=score,
                sources=sources,
                dense_rank=None if dense_hit is None else dense_hit.rank,
                dense_score=None if dense_hit is None else dense_hit.score,
                bm25_rank=None if bm25_hit is None else bm25_hit.rank,
                bm25_score=None if bm25_hit is None else bm25_hit.score,
                postgresql_version=source.postgresql_version,
                topic=source.topic,
                section_path=source.section_path,
                canonical_url=source.canonical_url,
                anchor=source.anchor,
                text=source.text,
            )
        )
    return output


def _direct_hits(
    hits: Sequence[DenseSearchHit] | Sequence[BM25SearchHit], route: str
) -> list[HybridSearchHit]:
    mapped = _route_map(hits, route)
    ordered = sorted(mapped.values(), key=lambda hit: (-hit.score, hit.chunk_id))
    output = []
    for rank, hit in enumerate(ordered, start=1):
        dense = hit if route == "dense" else None
        bm25 = hit if route == "bm25" else None
        output.append(
            HybridSearchHit(
                rank=rank,
                chunk_id=hit.chunk_id,
                score=hit.score,
                rrf_score=None,
                sources=(route,),
                dense_rank=None if dense is None else dense.rank,
                dense_score=None if dense is None else dense.score,
                bm25_rank=None if bm25 is None else bm25.rank,
                bm25_score=None if bm25 is None else bm25.score,
                postgresql_version=hit.postgresql_version,
                topic=hit.topic,
                section_path=hit.section_path,
                canonical_url=hit.canonical_url,
                anchor=hit.anchor,
                text=hit.text,
            )
        )
    return output


class HybridRetriever:
    """Lazy method router for lexical, dense, and fused retrieval."""

    def __init__(
        self,
        dense_factory: Callable[[], SearchRoute],
        bm25_factory: Callable[[], SearchRoute],
    ) -> None:
        self._dense_factory = dense_factory
        self._bm25_factory = bm25_factory
        self._dense: SearchRoute | None = None
        self._bm25: SearchRoute | None = None

    def _dense_route(self) -> SearchRoute:
        if self._dense is None:
            self._dense = self._dense_factory()
        return self._dense

    def _bm25_route(self) -> SearchRoute:
        if self._bm25 is None:
            self._bm25 = self._bm25_factory()
        return self._bm25

    async def search(
        self,
        query: str,
        version: str | None,
        top_k: int,
        *,
        method: str,
        dense_depth: int = 50,
        bm25_depth: int = 50,
        rrf_k: int = 60,
    ) -> list[HybridSearchHit]:
        if method not in {"b0", "b1", "m1"}:
            raise ValueError("method must be b0, b1, or m1")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if method == "b0":
            hits = await self._bm25_route().search(
                query, version, top_k, bm25_enabled=True
            )
            return _direct_hits(hits, "bm25")
        if method == "b1":
            hits = await self._dense_route().search(
                query, version, top_k, dense_enabled=True
            )
            return _direct_hits(hits, "dense")
        if type(dense_depth) is not int or dense_depth <= 0:
            raise ValueError("dense_depth must be a positive integer")
        if type(bm25_depth) is not int or bm25_depth <= 0:
            raise ValueError("bm25_depth must be a positive integer")
        if type(rrf_k) is not int or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        dense_hits = await self._dense_route().search(
            query, version, dense_depth, dense_enabled=True
        )
        bm25_hits = await self._bm25_route().search(
            query, version, bm25_depth, bm25_enabled=True
        )
        return reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=rrf_k, top_k=top_k)

    async def close(self) -> None:
        if self._dense is not None:
            await self._dense.close()
            self._dense = None
        if self._bm25 is not None:
            await self._bm25.close()
            self._bm25 = None
