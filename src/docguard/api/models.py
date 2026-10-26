"""Strict public schemas for the S9 HTTP API."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Version = Literal["17.11", "18.6"]
ActionName = Literal["answer", "refuse", "clarify", "error"]
ReasonName = Literal[
    "OUT_OF_DOMAIN",
    "NO_RELEVANT_EVIDENCE",
    "VERSION_NOT_AVAILABLE",
    "VERSION_AMBIGUOUS",
    "VERSION_CONFLICT",
    "EVIDENCE_INSUFFICIENT",
    "CITATION_VALIDATION_FAILED",
    "UPSTREAM_ERROR",
]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryRequest(StrictModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    requested_version: Version | None = None


class DecisionResponse(StrictModel):
    action: ActionName
    reason_code: ReasonName | None
    confidence: float = Field(ge=0.0, le=1.0)
    triggered_rule: NonEmpty
    feature_values: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reason(self) -> Self:
        if self.action == "answer" and self.reason_code is not None:
            raise ValueError("answer action cannot include a reason code")
        if self.action == "error" and self.reason_code != "UPSTREAM_ERROR":
            raise ValueError("error action requires UPSTREAM_ERROR")
        if self.action in {"refuse", "clarify"} and self.reason_code is None:
            raise ValueError("refuse and clarify actions require a reason code")
        return self


class ProvenanceResponse(StrictModel):
    sources: list[str]
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    rerank_truncated: bool | None = None
    original_rrf_rank: int | None = None


class EvidenceSummary(StrictModel):
    evidence_id: NonEmpty
    chunk_id: NonEmpty
    rank: int = Field(ge=1)
    score: float
    postgresql_version: Version
    topic: NonEmpty
    section_path: list[str]
    canonical_url: NonEmpty
    anchor: NonEmpty
    text: str
    provenance: ProvenanceResponse


class CitationResponse(StrictModel):
    evidence_id: NonEmpty
    chunk_id: NonEmpty
    postgresql_version: Version
    topic: NonEmpty
    section_path: list[str]
    canonical_url: NonEmpty
    anchor: NonEmpty
    text: str
    score: float


class ModelInfoResponse(StrictModel):
    model: str
    request_id: str | None = None
    provider_fingerprint: str | None = None
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    api_cost_usd: float = Field(ge=0.0)


class QueryResponse(StrictModel):
    request_id: NonEmpty
    question: NonEmpty
    requested_version: Version | None
    route: Literal["m3", "m4"]
    decision: DecisionResponse
    answer: str | None
    citations: list[CitationResponse]
    evidence: list[EvidenceSummary]
    latency_ms: dict[str, float]
    model_info: ModelInfoResponse | None

    @model_validator(mode="after")
    def validate_answer(self) -> Self:
        if self.decision.action == "answer" and not self.answer:
            raise ValueError("answer action requires a visible answer")
        if self.decision.action != "answer" and self.answer is not None:
            raise ValueError("non-answer action cannot include an answer")
        return self


class DependencyHealth(StrictModel):
    name: NonEmpty
    status: Literal["available", "unavailable"]
    detail: str | None = None


class HealthResponse(StrictModel):
    request_id: NonEmpty
    status: Literal["ok", "degraded"]
    dependencies: list[DependencyHealth]


class CorpusResponse(StrictModel):
    version: Version
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    index_status: Literal["ready", "unavailable"]
    snapshot_sha256: str | None = None
    built_at: str | None = None


class CorpusListResponse(StrictModel):
    request_id: NonEmpty
    corpora: list[CorpusResponse]


class RebuildResponse(StrictModel):
    request_id: NonEmpty
    job_id: NonEmpty
    status: Literal["queued"] = "queued"


class EvidenceResponse(StrictModel):
    request_id: NonEmpty
    chunk_id: NonEmpty
    postgresql_version: Version
    topic: NonEmpty
    section_path: list[str]
    canonical_url: NonEmpty
    anchor: NonEmpty
    text: str
    content_sha256: NonEmpty
    chunking_version: NonEmpty


class EvaluationCreateRequest(StrictModel):
    method: Literal["b0", "b1", "m1", "m2", "b2", "m4"]
    split: Literal["development", "validation"]


class EvaluationJobResponse(StrictModel):
    run_id: NonEmpty
    method: Literal["b0", "b1", "m1", "m2", "b2", "m4"]
    split: Literal["development", "validation"]
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: NonEmpty
    updated_at: NonEmpty
    metrics: dict[str, object] | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    @model_validator(mode="after")
    def validate_terminal_state(self) -> Self:
        if self.status == "failed" and not (self.failure_type and self.failure_message):
            raise ValueError("failed job requires failure information")
        if self.status != "failed" and (self.failure_type or self.failure_message):
            raise ValueError("only a failed job may include failure information")
        if self.status == "succeeded" and self.metrics is None:
            raise ValueError("succeeded job requires metrics")
        return self


class EvaluationListResponse(StrictModel):
    request_id: NonEmpty
    evaluations: list[EvaluationJobResponse]
