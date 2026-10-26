"""Application-level orchestration for one traceable M4 query."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from docguard.api.models import (
    CitationResponse,
    DecisionResponse,
    EvidenceSummary,
    ModelInfoResponse,
    ProvenanceResponse,
    QueryResponse,
)
from docguard.generation.models import GeneratedAnswer
from docguard.refusal.models import Action
from docguard.refusal.pipeline import M4Result


class M4QueryPipeline(Protocol):
    async def answer(
        self, question: str, requested_version: str | None, **switches: object
    ) -> M4Result: ...


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    response: QueryResponse
    http_status: int


def _provenance(value: Any) -> ProvenanceResponse:
    if hasattr(value, "provenance"):
        payload = value.provenance.to_dict()
    else:
        payload = {
            "sources": list(getattr(value, "sources", ())),
            "dense_rank": getattr(value, "dense_rank", None),
            "dense_score": getattr(value, "dense_score", None),
            "bm25_rank": getattr(value, "bm25_rank", None),
            "bm25_score": getattr(value, "bm25_score", None),
            "rrf_score": getattr(value, "rrf_score", None),
            "rerank_score": getattr(value, "rerank_score", None),
            "rerank_truncated": getattr(value, "rerank_truncated", None),
            "original_rrf_rank": getattr(value, "original_rrf_rank", None),
        }
    return ProvenanceResponse.model_validate(payload)


def _evidence(value: Any) -> EvidenceSummary:
    chunk_id = str(value.chunk_id)
    return EvidenceSummary(
        evidence_id=chunk_id,
        chunk_id=chunk_id,
        rank=int(value.rank),
        score=float(value.score),
        postgresql_version=value.postgresql_version,
        topic=value.topic,
        section_path=list(value.section_path),
        canonical_url=value.canonical_url,
        anchor=value.anchor,
        text=value.text,
        provenance=_provenance(value),
    )


def _citation(value: Any) -> CitationResponse:
    return CitationResponse(
        evidence_id=value.chunk_id,
        chunk_id=value.chunk_id,
        postgresql_version=value.postgresql_version,
        topic=value.topic,
        section_path=list(value.section_path),
        canonical_url=value.canonical_url,
        anchor=value.anchor,
        text=value.text,
        score=float(value.score),
    )


def _model_info(generated: GeneratedAnswer | None) -> ModelInfoResponse | None:
    if generated is None or generated.model is None:
        return None
    return ModelInfoResponse(
        model=generated.model,
        request_id=generated.request_id,
        provider_fingerprint=generated.provider_fingerprint,
        input_tokens=generated.usage.input_tokens,
        cached_input_tokens=generated.usage.cached_input_tokens,
        output_tokens=generated.usage.output_tokens,
        api_cost_usd=generated.usage.api_cost_usd,
    )


class QueryService:
    def __init__(
        self,
        pipeline: M4QueryPipeline,
        timeout_seconds: float,
        *,
        clock: Any = time.perf_counter,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._pipeline = pipeline
        self._timeout_seconds = timeout_seconds
        self._clock = clock

    async def query(
        self, question: str, requested_version: str | None, *, request_id: str
    ) -> QueryOutcome:
        started = self._clock()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await self._pipeline.answer(question, requested_version)
        except TimeoutError:
            elapsed = (self._clock() - started) * 1000.0
            response = QueryResponse(
                request_id=request_id,
                question=question,
                requested_version=requested_version,
                route="m4",
                decision=DecisionResponse(
                    action="error",
                    reason_code="UPSTREAM_ERROR",
                    confidence=1.0,
                    triggered_rule="request_timeout",
                    feature_values={},
                ),
                answer=None,
                citations=[],
                evidence=[],
                latency_ms={"end_to_end": elapsed},
                model_info=None,
            )
            return QueryOutcome(response, 504)

        generated = result.generated
        source_evidence: tuple[Any, ...]
        if generated is not None:
            source_evidence = tuple(generated.evidence)
        else:
            source_evidence = tuple(result.hits)
        latency = {"end_to_end": (self._clock() - started) * 1000.0}
        if generated is not None:
            latency.update(asdict(generated.timing_ms))
        decision = result.decision
        response = QueryResponse(
            request_id=request_id,
            question=question,
            requested_version=requested_version,
            route=result.route,
            decision=DecisionResponse(
                action=decision.action.value,
                reason_code=(decision.reason_code.value if decision.reason_code else None),
                confidence=decision.confidence,
                triggered_rule=decision.triggered_rule,
                feature_values=asdict(decision.features),
            ),
            answer=generated.answer if generated is not None else None,
            citations=(
                [_citation(item) for item in generated.citations]
                if generated is not None
                else []
            ),
            evidence=[_evidence(item) for item in source_evidence],
            latency_ms=latency,
            model_info=_model_info(generated),
        )
        status = 503 if decision.action is Action.ERROR else 200
        return QueryOutcome(response, status)

    async def close(self) -> None:
        close = getattr(self._pipeline, "close", None)
        if callable(close):
            await close()
