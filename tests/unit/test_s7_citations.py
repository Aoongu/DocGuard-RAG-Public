from dataclasses import replace

import pytest

from docguard.generation.citations import (
    GenerationValidationError,
    parse_generation_output,
    verify_citations,
)
from docguard.generation.models import Evidence, EvidenceProvenance


def _evidence(evidence_id: str = "E01") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id="chk_" + "a" * 64,
        rank=1,
        score=0.9,
        postgresql_version="17.11",
        topic="runtime-config",
        section_path=("Server Configuration", "Write Ahead Log"),
        canonical_url="https://www.postgresql.org/docs/17/runtime-config-wal.html",
        anchor="RUNTIME-CONFIG-WAL-SETTINGS",
        text="wal_level controls the amount of information written to WAL.",
        original_text_chars=61,
        text_truncated=False,
        provenance=EvidenceProvenance(
            sources=("dense",), dense_rank=1, dense_score=0.9
        ),
    )


def test_parse_and_verify_citations_maps_exact_source_metadata() -> None:
    parsed = parse_generation_output(
        '{"answer":"WAL level controls detail.","citations":["E01"]}'
    )
    verified = verify_citations(
        parsed, (_evidence(),), semantic_validation_enabled=True
    )

    assert parsed.answer == "WAL level controls detail."
    assert parsed.citation_ids == ("E01",)
    assert verified.answer == parsed.answer
    assert verified.validation_status == "verified"
    assert len(verified.citations) == 1
    citation = verified.citations[0]
    assert citation.evidence_id == "E01"
    assert citation.chunk_id == "chk_" + "a" * 64
    assert citation.postgresql_version == "17.11"
    assert citation.section_path == ("Server Configuration", "Write Ahead Log")
    assert citation.canonical_url.endswith("runtime-config-wal.html")
    assert citation.anchor == "RUNTIME-CONFIG-WAL-SETTINGS"
    assert citation.text.startswith("wal_level")
    assert citation.score == 0.9


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not-json", "valid JSON"),
        ("[]", "JSON object"),
        ('{"answer":"ok"}', "missing fields"),
        ('{"answer":"ok","citations":[],"extra":1}', "unknown fields"),
        ('{"answer":"  ","citations":[]}', "answer"),
        ('{"answer":1,"citations":[]}', "answer"),
        ('{"answer":"ok","citations":"E01"}', "citations"),
        ('{"answer":"ok","citations":[1]}', "citation ID"),
        ('{"answer":"ok","citations":["E1"]}', "citation ID"),
        ('{"answer":"ok","citations":["E01","E01"]}', "duplicate"),
    ],
)
def test_parse_generation_output_rejects_invalid_shapes(raw: str, message: str) -> None:
    with pytest.raises(GenerationValidationError, match=message):
        parse_generation_output(raw)


def test_parse_generation_output_rejects_oversized_or_unsafe_answer() -> None:
    with pytest.raises(GenerationValidationError, match="answer"):
        parse_generation_output(
            '{"answer":"' + "a" * 8_001 + '","citations":[]}'
        )
    with pytest.raises(GenerationValidationError, match="answer"):
        parse_generation_output('{"answer":"bad\\u0000text","citations":[]}')


def test_verify_citations_always_rejects_unknown_ids() -> None:
    parsed = parse_generation_output('{"answer":"Unsupported.","citations":["E02"]}')
    for semantic_validation_enabled in (True, False):
        with pytest.raises(GenerationValidationError, match="unknown evidence_id"):
            verify_citations(
                parsed,
                (_evidence(),),
                semantic_validation_enabled=semantic_validation_enabled,
            )


def test_semantic_validation_requires_at_least_one_citation() -> None:
    parsed = parse_generation_output('{"answer":"Insufficient evidence.","citations":[]}')
    with pytest.raises(GenerationValidationError, match="at least one citation"):
        verify_citations(parsed, (_evidence(),), semantic_validation_enabled=True)


def test_structural_only_mode_allows_empty_known_id_list() -> None:
    parsed = parse_generation_output('{"answer":"Insufficient evidence.","citations":[]}')
    verified = verify_citations(
        parsed, (_evidence(),), semantic_validation_enabled=False
    )
    assert verified.answer == "Insufficient evidence."
    assert verified.citations == ()
    assert verified.validation_status == "structural_only"


def test_verify_citations_rejects_duplicate_evidence_allow_list() -> None:
    parsed = parse_generation_output('{"answer":"Answer.","citations":["E01"]}')
    with pytest.raises(GenerationValidationError, match="duplicate evidence_id"):
        verify_citations(
            parsed,
            (_evidence(), replace(_evidence(), chunk_id="chk_" + "b" * 64)),
            semantic_validation_enabled=True,
        )
