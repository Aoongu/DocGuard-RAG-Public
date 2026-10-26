from __future__ import annotations

from pathlib import Path

import pytest

from docguard.generation.citations import GenerationValidationError
from docguard.generation.models import (
    GeneratedAnswer,
    GenerationFailureContext,
    GenerationTiming,
    GenerationUsage,
)
from docguard.generation.service import PostResponseGenerationError
from docguard.refusal.config import load_refusal_config
from docguard.refusal.models import Action, RankedEvidence, ReasonCode
from docguard.refusal.pipeline import M4Pipeline
from docguard.refusal.policy import RefusalThresholds

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_refusal_config(ROOT / "configs" / "refusal" / "s8-refusal-v1.json")
THRESHOLDS = RefusalThresholds(top1=0.0, top3_mean=0.0, coverage=0.75, margin=0.5)


def _hit(rank: int, score: float, version: str = "17.11") -> RankedEvidence:
    return RankedEvidence(
        rank=rank,
        chunk_id=f"c-{version}-{rank}",
        rerank_score=score,
        postgresql_version=version,
        canonical_url=f"https://www.postgresql.org/docs/{version.split('.')[0]}/sql-select.html",
        anchor=f"A{rank}",
        text="SELECT reads rows from a PostgreSQL table and returns result rows.",
    )


class _Retriever:
    def __init__(
        self, hits: list[RankedEvidence], error: Exception | None = None
    ) -> None:
        self.hits = hits
        self.error = error
        self.calls: list[tuple[object, ...]] = []
        self.closed = 0

    async def search(
        self, query: str, version: str | None, top_k: int, **options: object
    ):
        self.calls.append((query, version, top_k, options))
        if self.error is not None:
            raise self.error
        return self.hits

    async def close(self) -> None:
        self.closed += 1


class _TimedRetriever(_Retriever):
    def consume_timing_ms(self) -> dict[str, float]:
        return {"retrieval": 12.0, "reranking": 3.0}


class _Generator:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.closed = 0
        self.error = error

    async def answer(self, question, requested_version, hits, **options):
        self.calls.append((question, requested_version, tuple(hits), options))
        if self.error is not None:
            raise self.error
        return GeneratedAnswer(
            status="success",
            answer="answer",
            citations=(),
            evidence=(),
            validation_status="structural_only",
            raw_content=None,
            request_id="r",
            model="m",
            usage=GenerationUsage(0, 0, 0, 0.0),
            timing_ms=GenerationTiming(0.0, 0.0, 0.0, 0.0),
        )

    async def close(self) -> None:
        self.closed += 1


@pytest.mark.asyncio
async def test_unavailable_version_refuses_without_retrieval_or_generation() -> None:
    retriever = _Retriever([])
    generator = _Generator()
    result = await M4Pipeline(retriever, generator, CONFIG, THRESHOLDS).answer(
        "According to PostgreSQL 9.0, what does SELECT do?", None
    )
    assert (result.decision.action, result.decision.reason_code) == (
        Action.REFUSE,
        ReasonCode.VERSION_NOT_AVAILABLE,
    )
    assert retriever.calls == []
    assert generator.calls == []


@pytest.mark.asyncio
async def test_refuse_and_clarify_never_call_generation() -> None:
    weak = _Retriever([_hit(1, -3.0), _hit(2, -3.0), _hit(3, -3.0)])
    generator = _Generator()
    refused = await M4Pipeline(weak, generator, CONFIG, THRESHOLDS).answer(
        "What does PostgreSQL SELECT do?", None
    )
    assert refused.decision.action is Action.REFUSE
    assert generator.calls == []


@pytest.mark.asyncio
async def test_answer_passes_only_requested_version_to_s7() -> None:
    retriever = _Retriever([_hit(1, 2.0), _hit(2, 1.0, "18.6"), _hit(3, 0.5)])
    generator = _Generator()
    result = await M4Pipeline(retriever, generator, CONFIG, THRESHOLDS).answer(
        "According to PostgreSQL 17.11, what does SELECT do?", "17.11"
    )
    assert result.decision.action is Action.ANSWER
    assert retriever.calls[0][1] == "17.11"
    assert len(generator.calls) == 1
    assert {hit.postgresql_version for hit in generator.calls[0][2]} == {"17.11"}
    assert generator.calls[0][3]["citation_validation_enabled"] is True
    assert result.route == "m4"
    assert dict(result.switches) == {
        "citation_validation_enabled": True,
        "generation_enabled": True,
        "refusal_enabled": True,
        "version_filter_enabled": True,
    }


@pytest.mark.asyncio
async def test_answer_preserves_retrieval_and_reranking_timing() -> None:
    retriever = _TimedRetriever([_hit(1, 2.0), _hit(2, 1.0), _hit(3, 0.5)])
    result = await M4Pipeline(retriever, _Generator(), CONFIG, THRESHOLDS).answer(
        "What does PostgreSQL SELECT do?", None
    )

    assert dict(result.timing_ms) == {"retrieval": 12.0, "reranking": 3.0}


@pytest.mark.asyncio
async def test_upstream_error_is_distinct_and_pipeline_closes_once() -> None:
    retriever = _Retriever([], RuntimeError("timeout"))
    generator = _Generator()
    pipeline = M4Pipeline(retriever, generator, CONFIG, THRESHOLDS)
    result = await pipeline.answer("What does PostgreSQL SELECT do?", None)
    assert (result.decision.action, result.decision.reason_code) == (
        Action.ERROR,
        ReasonCode.UPSTREAM_ERROR,
    )
    assert generator.calls == []
    await pipeline.close()
    await pipeline.close()
    assert retriever.closed == generator.closed == 1


@pytest.mark.asyncio
async def test_post_generation_citation_failure_becomes_refusal_not_upstream_error() -> (
    None
):
    context = GenerationFailureContext(
        raw_content="{}",
        evidence=(),
        request_id="r",
        model="m",
        usage=GenerationUsage(0, 0, 0, 0.0),
        timing_ms=GenerationTiming(0.0, 0.0, 0.0, 0.0),
    )
    error = PostResponseGenerationError(
        GenerationValidationError("unknown citation"), context
    )
    generator = _Generator(error)
    retriever = _Retriever([_hit(1, 2.0), _hit(2, 1.0), _hit(3, 0.5)])
    result = await M4Pipeline(retriever, generator, CONFIG, THRESHOLDS).answer(
        "According to PostgreSQL 17.11, what does SELECT do?", "17.11"
    )
    assert (result.decision.action, result.decision.reason_code) == (
        Action.REFUSE,
        ReasonCode.CITATION_VALIDATION_FAILED,
    )
