import json
from pathlib import Path

import pytest

from docguard.evaluation.config import load_evaluation_config
from docguard.evaluation.models import EvaluationQuestion, SourceReference


def _question_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "question_id": "s4-dev-001",
        "question": "How is VACUUM used in PostgreSQL 17.11?",
        "requested_version": "17.11",
        "answerable": True,
        "answer_key_points": ["VACUUM reclaims or reuses storage."],
        "relevant_source_refs": [
            {
                "version": "17.11",
                "normalized_url": "https://www.postgresql.org/docs/17/sql-vacuum.html",
                "anchor": "SQL-VACUUM",
            }
        ],
        "resolved_chunk_ids": ["chk_" + "a" * 64],
        "chunking_version": "s2-chunk-v1",
        "expected_reason_code": None,
        "category": "how_to_parameter",
        "split": "development",
        "source": "assistant_curated",
        "review_status": "source_verified",
        "paraphrase_group_id": "vacuum-purpose",
    }


def test_question_round_trip_is_strict() -> None:
    payload = _question_payload()
    question = EvaluationQuestion.from_dict(payload)

    assert question.to_dict() == payload
    with pytest.raises(ValueError, match="unknown fields"):
        EvaluationQuestion.from_dict({**payload, "surprise": True})
    incomplete = dict(payload)
    incomplete.pop("question")
    with pytest.raises(ValueError, match="missing fields"):
        EvaluationQuestion.from_dict(incomplete)


def test_answerability_contract_rejects_inconsistent_reason_and_evidence() -> None:
    payload = _question_payload()
    with pytest.raises(ValueError, match="reason code"):
        EvaluationQuestion.from_dict(
            {**payload, "expected_reason_code": "NO_EVIDENCE"}
        )

    boundary = {
        **payload,
        "question_id": "s4-dev-031",
        "answerable": False,
        "answer_key_points": [],
        "relevant_source_refs": [],
        "resolved_chunk_ids": [],
        "expected_reason_code": "OUT_OF_DOMAIN",
        "category": "out_of_domain",
    }
    assert EvaluationQuestion.from_dict(boundary).expected_reason_code == "OUT_OF_DOMAIN"
    with pytest.raises(ValueError, match="must not contain evidence"):
        EvaluationQuestion.from_dict(
            {**boundary, "resolved_chunk_ids": ["chk_" + "b" * 64]}
        )


def test_source_reference_rejects_non_official_or_version_mismatched_url() -> None:
    with pytest.raises(ValueError, match="official versioned"):
        SourceReference.from_dict(
            {
                "version": "18.6",
                "normalized_url": "https://example.com/docs/18/sql-select.html",
                "anchor": "SQL-SELECT",
            }
        )
    with pytest.raises(ValueError, match="official versioned"):
        SourceReference.from_dict(
            {
                "version": "18.6",
                "normalized_url": "https://www.postgresql.org/docs/17/sql-select.html",
                "anchor": "SQL-SELECT",
            }
        )


def test_load_frozen_evaluation_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    payload = {
        "schema_version": 1,
        "evaluation_version": "s4-eval-v1",
        "dataset_version": "s4-golden-v1",
        "allowed_splits": ["development", "validation"],
        "retrieval_cutoffs": [1, 5, 10],
        "mrr_cutoff": 10,
        "ndcg_cutoff": 10,
        "retrieval_top_k": 10,
        "bootstrap_seed": 20261014,
        "bootstrap_samples": 10000,
        "confidence_level": 0.95,
        "chunking_version": "s2-chunk-v1",
        "corpus_manifest_sha256": "0" * 64,
        "chunks_sha256": "1" * 64,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_evaluation_config(path)

    assert config.retrieval_cutoffs == (1, 5, 10)
    assert "test" not in config.allowed_splits
    with pytest.raises(ValueError, match="allowed_splits"):
        path.write_text(
            json.dumps({**payload, "allowed_splits": ["development", "test"]}),
            encoding="utf-8",
        )
        load_evaluation_config(path)
