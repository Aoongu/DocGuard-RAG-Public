"""Strict local parsing and citation allow-list validation."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from docguard.generation.models import (
    Evidence,
    ParsedGenerationOutput,
    VerifiedCitation,
    VerifiedGeneration,
)

_FIELDS = frozenset({"answer", "citations"})
_EVIDENCE_ID_RE = re.compile(r"E(?:0[1-9]|10)")


class GenerationValidationError(RuntimeError):
    """Raised when model output cannot become a verified answer."""


def parse_generation_output(raw: str) -> ParsedGenerationOutput:
    if not isinstance(raw, str):
        raise GenerationValidationError("generation output must be text")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise GenerationValidationError("generation output must be valid JSON") from error
    if not isinstance(payload, dict):
        raise GenerationValidationError("generation output must be a JSON object")
    unknown = set(payload) - _FIELDS
    missing = _FIELDS - set(payload)
    if unknown:
        raise GenerationValidationError(
            f"generation output contains unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise GenerationValidationError(
            f"generation output is missing fields: {sorted(missing)}"
        )
    answer = payload["answer"]
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or len(answer) > 8_000
        or "\0" in answer
    ):
        raise GenerationValidationError("answer must be non-empty safe text up to 8000 chars")
    raw_citations = payload["citations"]
    if not isinstance(raw_citations, list) or len(raw_citations) > 10:
        raise GenerationValidationError("citations must be a list of at most 10 IDs")
    citations: list[str] = []
    for item in raw_citations:
        if not isinstance(item, str) or _EVIDENCE_ID_RE.fullmatch(item) is None:
            raise GenerationValidationError("citation ID must match E01 through E10")
        if item in citations:
            raise GenerationValidationError(f"duplicate citation ID: {item}")
        citations.append(item)
    return ParsedGenerationOutput(answer=answer.strip(), citation_ids=tuple(citations))


def verify_citations(
    parsed: ParsedGenerationOutput,
    evidence: Sequence[Evidence],
    *,
    semantic_validation_enabled: bool,
) -> VerifiedGeneration:
    if type(semantic_validation_enabled) is not bool:
        raise TypeError("semantic_validation_enabled must be boolean")
    evidence_map: dict[str, Evidence] = {}
    for item in evidence:
        if item.evidence_id in evidence_map:
            raise GenerationValidationError(
                f"duplicate evidence_id in allow-list: {item.evidence_id}"
            )
        evidence_map[item.evidence_id] = item
    if semantic_validation_enabled and not parsed.citation_ids:
        raise GenerationValidationError("verified answer requires at least one citation")

    citations: list[VerifiedCitation] = []
    for evidence_id in parsed.citation_ids:
        item = evidence_map.get(evidence_id)
        if item is None:
            raise GenerationValidationError(f"unknown evidence_id: {evidence_id}")
        citations.append(
            VerifiedCitation(
                evidence_id=item.evidence_id,
                chunk_id=item.chunk_id,
                postgresql_version=item.postgresql_version,
                topic=item.topic,
                section_path=item.section_path,
                canonical_url=item.canonical_url,
                anchor=item.anchor,
                text=item.text,
                score=item.score,
            )
        )
    return VerifiedGeneration(
        answer=parsed.answer,
        citations=tuple(citations),
        validation_status=("verified" if semantic_validation_enabled else "structural_only"),
    )
