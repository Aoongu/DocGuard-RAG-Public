"""Immutable records shared by S7 generation components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    messages: tuple[dict[str, str], ...]
    output_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GenerationCompletion:
    content: str
    request_id: str
    model: str
    finish_reason: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    provider_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedGenerationOutput:
    answer: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifiedCitation:
    evidence_id: str
    chunk_id: str
    postgresql_version: str
    topic: str
    section_path: tuple[str, ...]
    canonical_url: str
    anchor: str
    text: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "postgresql_version": self.postgresql_version,
            "topic": self.topic,
            "section_path": list(self.section_path),
            "canonical_url": self.canonical_url,
            "anchor": self.anchor,
            "text": self.text,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class VerifiedGeneration:
    answer: str
    citations: tuple[VerifiedCitation, ...]
    validation_status: str


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    api_cost_usd: float
    api_cost_usd_min: float | None = None
    api_cost_usd_max: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationTiming:
    evidence: float
    generation: float
    validation: float
    end_to_end: float
    retrieval: float = 0.0


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    status: str
    answer: str | None
    citations: tuple[VerifiedCitation, ...]
    evidence: tuple[Evidence, ...]
    validation_status: str | None
    raw_content: str | None
    request_id: str | None
    model: str | None
    usage: GenerationUsage
    timing_ms: GenerationTiming
    provider_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationFailureContext:
    raw_content: str
    evidence: tuple[Evidence, ...]
    request_id: str
    model: str
    usage: GenerationUsage
    timing_ms: GenerationTiming
    provider_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    sources: tuple[str, ...]
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    rerank_truncated: bool | None = None
    original_rrf_rank: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "sources": list(self.sources),
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            "rerank_truncated": self.rerank_truncated,
            "original_rrf_rank": self.original_rrf_rank,
        }


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    chunk_id: str
    rank: int
    score: float
    postgresql_version: str
    topic: str
    section_path: tuple[str, ...]
    canonical_url: str
    anchor: str
    text: str
    original_text_chars: int
    text_truncated: bool
    provenance: EvidenceProvenance

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "postgresql_version": self.postgresql_version,
            "topic": self.topic,
            "section_path": list(self.section_path),
            "canonical_url": self.canonical_url,
            "anchor": self.anchor,
            "text": self.text,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.to_prompt_dict(),
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "score": self.score,
            "original_text_chars": self.original_text_chars,
            "text_truncated": self.text_truncated,
            "provenance": self.provenance.to_dict(),
        }
