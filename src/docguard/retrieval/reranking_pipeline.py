"""Lazy M2 orchestration for the frozen S5 candidate route and S6 reranker."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from docguard.retrieval.hybrid import HybridSearchHit
from docguard.retrieval.reranker import PairScorer, RerankedSearchHit, rerank_hits
from docguard.retrieval.reranker_config import RerankerConfig


class HybridRoute(Protocol):
    async def search(
        self, query: str, version: str | None, top_k: int, **options: object
    ) -> list[HybridSearchHit]: ...

    async def close(self) -> None: ...


class RerankingRetriever:
    """Retrieve M1 candidates and lazily apply the fixed Cross-Encoder."""

    def __init__(
        self,
        hybrid: HybridRoute,
        reranker_factory: Callable[[], PairScorer],
        config: RerankerConfig,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._hybrid = hybrid
        self._reranker_factory = reranker_factory
        self._config = config
        self._clock = clock
        self._reranker: PairScorer | None = None
        self._last_timing_ms: dict[str, float] | None = None

    def _get_reranker(self) -> PairScorer:
        if self._reranker is None:
            self._reranker = self._reranker_factory()
        return self._reranker

    async def search(
        self,
        query: str,
        version: str | None,
        top_k: int,
        *,
        method: str,
        rerank_depth: int = 20,
        rerank_enabled: bool = True,
    ) -> list[HybridSearchHit] | list[RerankedSearchHit]:
        if method != "m2":
            raise ValueError("method must be m2")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if rerank_depth not in self._config.rerank_depth_candidates:
            raise ValueError("rerank_depth is outside the frozen candidate set")
        if top_k > rerank_depth:
            raise ValueError("top_k cannot exceed rerank_depth")
        if type(rerank_enabled) is not bool:
            raise TypeError("rerank_enabled must be boolean")

        started = self._clock()
        candidates = await self._hybrid.search(
            query,
            version,
            rerank_depth if rerank_enabled else top_k,
            method="m1",
            dense_depth=self._config.dense_depth,
            bm25_depth=self._config.bm25_depth,
            rrf_k=self._config.rrf_k,
        )
        retrieved = self._clock()
        if not rerank_enabled:
            self._last_timing_ms = {
                "retrieval": (retrieved - started) * 1000.0,
                "reranking": 0.0,
            }
            return candidates

        hits = rerank_hits(query, candidates, self._get_reranker(), top_k=top_k)
        ended = self._clock()
        self._last_timing_ms = {
            "retrieval": (retrieved - started) * 1000.0,
            "reranking": (ended - retrieved) * 1000.0,
        }
        return hits

    def consume_timing_ms(self) -> dict[str, float] | None:
        timing = self._last_timing_ms
        self._last_timing_ms = None
        return None if timing is None else dict(timing)

    async def close(self) -> None:
        if self._reranker is not None:
            close = getattr(self._reranker, "close", None)
            if callable(close):
                close()
            self._reranker = None
        await self._hybrid.close()
