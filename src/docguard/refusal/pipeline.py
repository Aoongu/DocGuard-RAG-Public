"""M4 orchestration: M2, pre-generation refusal, then S7 generation."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from docguard.generation.models import GeneratedAnswer
from docguard.generation.service import PostResponseGenerationError
from docguard.refusal.config import RefusalConfig
from docguard.refusal.features import extract_refusal_features
from docguard.refusal.models import (
    Action,
    AnswerDecision,
    ReasonCode,
    RefusalFeatures,
)
from docguard.refusal.policy import RefusalThresholds, decide
from docguard.refusal.versioning import detect_version_conflict, parse_version_request

_UPSTREAM_ERRORS = (RuntimeError, ValueError, TypeError, OSError)


class M2Route(Protocol):
    async def search(
        self, query: str, version: str | None, top_k: int, **options: object
    ) -> Sequence[Any]: ...

    async def close(self) -> None: ...


class AnswerGenerator(Protocol):
    async def answer(
        self,
        question: str,
        requested_version: str | None,
        hits: Sequence[Any],
        *,
        generation_enabled: bool,
        citation_validation_enabled: bool,
    ) -> GeneratedAnswer: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class M4Result:
    route: str
    decision: AnswerDecision
    generated: GeneratedAnswer | None
    hits: tuple[Any, ...]
    error_message: str | None = None
    switches: tuple[tuple[str, bool], ...] = ()
    timing_ms: tuple[tuple[str, float], ...] = ()


def _empty_features(requested_version: str | None, status: str) -> RefusalFeatures:
    return RefusalFeatures(
        top1_rerank_score=0.0,
        top1_top2_margin=0.0,
        top3_mean_rerank_score=0.0,
        evidence_term_coverage=0.0,
        requested_version=requested_version,
        requested_version_status=status,
        top_evidence_versions=(),
        best_score_by_version=(),
        paired_version_conflict=False,
        postgresql_domain_signal=False,
    )


class M4Pipeline:
    def __init__(
        self,
        retriever: M2Route,
        generation: AnswerGenerator,
        config: RefusalConfig,
        thresholds: RefusalThresholds,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._retriever = retriever
        self._generation = generation
        self._config = config
        self._thresholds = thresholds
        self._clock = clock
        self._closed = False

    def _retrieval_timing(self, started: float) -> tuple[tuple[str, float], ...]:
        consume = getattr(self._retriever, "consume_timing_ms", None)
        supplied = consume() if callable(consume) else None
        if supplied is None:
            supplied = {"retrieval": (self._clock() - started) * 1000.0}
        return tuple(
            sorted((str(key), float(value)) for key, value in supplied.items())
        )

    async def answer(
        self,
        question: str,
        requested_version: str | None,
        *,
        refusal_enabled: bool = True,
        version_filter_enabled: bool = True,
        generation_enabled: bool = True,
        citation_validation_enabled: bool = True,
    ) -> M4Result:
        route = "m4" if refusal_enabled else "m3"
        switches = tuple(
            sorted(
                {
                    "refusal_enabled": refusal_enabled,
                    "version_filter_enabled": version_filter_enabled,
                    "generation_enabled": generation_enabled,
                    "citation_validation_enabled": citation_validation_enabled,
                }.items()
            )
        )
        version_request = parse_version_request(
            question, requested_version, self._config
        )
        if version_request.status == "unavailable":
            features = _empty_features(
                version_request.requested_version, version_request.status
            )
            decision = decide(features, version_request, self._thresholds)
            return M4Result(route, decision, None, (), switches=switches)

        retrieval_version = (
            version_request.requested_version
            if version_filter_enabled and version_request.status == "available"
            else None
        )
        retrieval_started = self._clock()
        try:
            raw_hits = list(
                await self._retriever.search(
                    question.strip(),
                    retrieval_version,
                    10,
                    method="m2",
                    rerank_depth=50,
                    rerank_enabled=True,
                )
            )
        except _UPSTREAM_ERRORS as error:
            features = _empty_features(
                version_request.requested_version, version_request.status
            )
            decision = decide(
                features,
                version_request,
                self._thresholds,
                upstream_error=type(error).__name__,
            )
            return M4Result(
                route,
                decision,
                None,
                (),
                str(error),
                switches=switches,
                timing_ms=(
                    ("retrieval", (self._clock() - retrieval_started) * 1000.0),
                ),
            )
        retrieval_timing = self._retrieval_timing(retrieval_started)

        if version_filter_enabled and version_request.status == "available":
            raw_hits = [
                hit
                for hit in raw_hits
                if hit.postgresql_version == version_request.requested_version
            ]
        normalized = [replace(hit, rank=index) for index, hit in enumerate(raw_hits, 1)]
        conflict = detect_version_conflict(normalized, self._config)
        features = extract_refusal_features(
            question,
            normalized,
            self._config,
            requested_version=version_request.requested_version,
            requested_version_status=version_request.status,
            paired_version_conflict=conflict.detected,
        )
        decision = decide(
            features,
            version_request,
            self._thresholds,
            refusal_enabled=refusal_enabled,
            version_filter_enabled=version_filter_enabled,
        )
        if decision.action is not Action.ANSWER:
            return M4Result(
                route,
                decision,
                None,
                tuple(normalized),
                switches=switches,
                timing_ms=retrieval_timing,
            )
        try:
            generated = await self._generation.answer(
                question.strip(),
                version_request.requested_version,
                normalized,
                generation_enabled=generation_enabled,
                citation_validation_enabled=citation_validation_enabled,
            )
        except PostResponseGenerationError as error:
            failed = AnswerDecision(
                Action.REFUSE,
                ReasonCode.CITATION_VALIDATION_FAILED,
                1.0,
                "post_generation_citation_validation_failed",
                features,
            )
            return M4Result(
                route,
                failed,
                None,
                tuple(normalized),
                str(error),
                switches=switches,
                timing_ms=retrieval_timing,
            )
        except _UPSTREAM_ERRORS as error:
            failed = AnswerDecision(
                Action.ERROR,
                ReasonCode.UPSTREAM_ERROR,
                1.0,
                "generation_upstream_error",
                features,
            )
            return M4Result(
                route,
                failed,
                None,
                tuple(normalized),
                str(error),
                switches=switches,
                timing_ms=retrieval_timing,
            )
        return M4Result(
            route,
            decision,
            generated,
            tuple(normalized),
            switches=switches,
            timing_ms=retrieval_timing,
        )

    async def close(self) -> None:
        if not self._closed:
            await self._generation.close()
            await self._retriever.close()
            self._closed = True
