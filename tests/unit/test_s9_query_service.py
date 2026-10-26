from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from docguard.generation.models import (
    Evidence,
    EvidenceProvenance,
    GeneratedAnswer,
    GenerationTiming,
    GenerationUsage,
    VerifiedCitation,
)
from docguard.refusal.models import Action, AnswerDecision, ReasonCode, RefusalFeatures
from docguard.refusal.pipeline import M4Result
from docguard.service.query import QueryService


def _features() -> RefusalFeatures:
    return RefusalFeatures(1.0, 0.5, 0.8, 0.9, "18.6", "available", ("18.6",), (), False, True)


@dataclass(frozen=True)
class _Hit:
    rank: int = 1
    chunk_id: str = "chk_18"
    score: float = 1.0
    rerank_score: float = 1.0
    rerank_truncated: bool = False
    original_rrf_rank: int = 2
    rrf_score: float = 0.03
    sources: tuple[str, ...] = ("dense", "bm25")
    dense_rank: int = 1
    dense_score: float = 0.8
    bm25_rank: int = 3
    bm25_score: float = 2.0
    postgresql_version: str = "18.6"
    topic: str = "admin"
    section_path: tuple[str, ...] = ("Reference", "VACUUM")
    canonical_url: str = "https://example/18"
    anchor: str = "vacuum"
    text: str = "VACUUM reclaims storage."


def _generated() -> GeneratedAnswer:
    provenance = EvidenceProvenance(
        sources=("dense", "bm25"), dense_rank=1, bm25_rank=3, rerank_score=1.0
    )
    evidence = Evidence(
        "E01", "chk_18", 1, 1.0, "18.6", "admin", ("Reference", "VACUUM"),
        "https://example/18", "vacuum", "VACUUM reclaims storage.", 24, False, provenance
    )
    citation = VerifiedCitation(
        "E01", "chk_18", "18.6", "admin", ("Reference", "VACUUM"),
        "https://example/18", "vacuum", "VACUUM reclaims storage.", 1.0
    )
    return GeneratedAnswer(
        "success", "VACUUM reclaims storage.", (citation,), (evidence,), "verified", None,
        "provider-req", "deepseek-flash", GenerationUsage(10, 0, 5, 0.001),
        GenerationTiming(1.0, 2.0, 1.0, 5.0, 1.0), None
    )


class _Pipeline:
    def __init__(self, result: M4Result, delay: float = 0.0) -> None:
        self.result = result
        self.delay = delay

    async def answer(self, question: str, requested_version: str | None, **_: object) -> M4Result:
        await asyncio.sleep(self.delay)
        return self.result


@pytest.mark.asyncio
async def test_answer_maps_public_evidence_ids_to_stable_chunk_ids() -> None:
    result = M4Result(
        "m4", AnswerDecision(Action.ANSWER, None, 0.8, "evidence_sufficient", _features()),
        _generated(), (_Hit(),)
    )

    outcome = await QueryService(_Pipeline(result), timeout_seconds=1).query(
        "VACUUM?", "18.6", request_id="req-1"
    )

    assert outcome.http_status == 200
    assert outcome.response.answer == "VACUUM reclaims storage."
    assert outcome.response.citations[0].evidence_id == "chk_18"
    assert outcome.response.evidence[0].evidence_id == "chk_18"


@pytest.mark.asyncio
async def test_refusal_is_successful_http_response() -> None:
    decision = AnswerDecision(
        Action.REFUSE, ReasonCode.NO_RELEVANT_EVIDENCE, 0.9, "below_threshold", _features()
    )

    outcome = await QueryService(_Pipeline(M4Result("m4", decision, None, ())), 1).query(
        "unknown", None, request_id="req-2"
    )

    assert outcome.http_status == 200
    assert outcome.response.decision.action == "refuse"
    assert outcome.response.answer is None


@pytest.mark.asyncio
async def test_upstream_error_is_not_knowledge_refusal() -> None:
    decision = AnswerDecision(
        Action.ERROR, ReasonCode.UPSTREAM_ERROR, 1.0, "upstream_error", _features()
    )

    outcome = await QueryService(_Pipeline(M4Result("m4", decision, None, (), "secret")), 1).query(
        "question", None, request_id="req-3"
    )

    assert outcome.http_status == 503
    assert outcome.response.decision.action == "error"
    assert outcome.response.decision.reason_code == "UPSTREAM_ERROR"
    assert "secret" not in outcome.response.model_dump_json()


@pytest.mark.asyncio
async def test_timeout_returns_upstream_error_with_504() -> None:
    decision = AnswerDecision(Action.ANSWER, None, 0.8, "ok", _features())
    service = QueryService(_Pipeline(M4Result("m4", decision, _generated(), ()), 0.05), 0.001)

    outcome = await service.query("question", None, request_id="req-4")

    assert outcome.http_status == 504
    assert outcome.response.decision.reason_code == "UPSTREAM_ERROR"
    assert outcome.response.decision.triggered_rule == "request_timeout"
