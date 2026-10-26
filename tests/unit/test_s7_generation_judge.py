import asyncio
import json
from pathlib import Path

import pytest

from docguard.evaluation.generation_judge import (
    HumanReviewError,
    compare_judge_to_human,
    create_review_template,
    judge_output_schema,
    judge_output_schema_sha256,
    judge_prompt_sha256,
    parse_judge_output,
    run_judge_evaluation,
    select_review_sample,
    summarize_human_review,
)
from docguard.evaluation.models import EvaluationQuestion
from docguard.evaluation.runner import TestSplitLockedError
from docguard.generation.client import FixedGenerationClient
from docguard.generation.config import load_generation_config
from docguard.generation.models import GenerationCompletion


def _question(number: int, category: str, answerable: bool) -> EvaluationQuestion:
    chunk = "chk_" + f"{number:064x}"
    return EvaluationQuestion.from_dict(
        {
            "schema_version": 1,
            "question_id": f"s4-dev-{number:03d}",
            "question": f"Question {number}?",
            "requested_version": "18.6",
            "answerable": answerable,
            "answer_key_points": ["fact"] if answerable else [],
            "relevant_source_refs": (
                [
                    {
                        "version": "18.6",
                        "normalized_url": "https://www.postgresql.org/docs/18/sql-select.html",
                        "anchor": f"SELECT-{number}",
                    }
                ]
                if answerable
                else []
            ),
            "resolved_chunk_ids": [chunk] if answerable else [],
            "chunking_version": "s2-chunk-v1",
            "expected_reason_code": None if answerable else "NO_EVIDENCE",
            "category": category,
            "split": "development",
            "source": "human_written",
            "review_status": "human_reviewed",
            "paraphrase_group_id": f"group-{number}",
        }
    )


def test_parse_judge_output_is_strict() -> None:
    parsed = parse_judge_output(
        '{"faithfulness_score":2,"answer_relevance_score":1,"rationale":"Supported."}'
    )
    assert parsed.faithfulness_score == 2
    assert parsed.answer_relevance_score == 1
    with pytest.raises(ValueError, match="unknown"):
        parse_judge_output(
            '{"faithfulness_score":2,"answer_relevance_score":1,'
            '"rationale":"ok","extra":true}'
        )


def test_judge_prompt_and_schema_identities_are_stable() -> None:
    assert len(judge_prompt_sha256()) == 64
    assert len(judge_output_schema_sha256()) == 64
    assert judge_output_schema()["required"] == [
        "faithfulness_score",
        "answer_relevance_score",
        "rationale",
    ]


def test_judge_runner_is_resumable_and_reports_human_intersection(
    tmp_path: Path,
) -> None:
    generation_rows = [
        {
            "schema_version": 1,
            "question_id": "s4-dev-001",
            "question": "What is supported?",
            "split": "development",
            "status": "success",
            "answer": "Supported answer.",
            "citations": [
                {
                    "evidence_id": "E01",
                    "chunk_id": "chk_" + "1" * 64,
                    "postgresql_version": "18.6",
                    "topic": "sql_commands",
                    "section_path": ["Description"],
                    "canonical_url": "https://www.postgresql.org/docs/18/sql-select.html",
                    "anchor": "description",
                    "text": "Supported answer.",
                    "score": 0.9,
                }
            ],
        },
        {
            "schema_version": 1,
            "question_id": "s4-dev-002",
            "question": "Failed?",
            "split": "development",
            "status": "failed",
            "answer": None,
            "citations": [],
        },
    ]
    human_review = {
        "schema_version": 2,
        "human_validation_status": "complete",
        "judge_prompt_version": "s7-judge-v1",
        "rows": [
            {
                "question_id": "s4-dev-001",
                "generation_status": "success",
                "visible_answer": "Supported answer.",
                "review_status": "evaluable",
                "human_faithfulness_score": 2,
                "human_answer_relevance_score": 2,
                "primary_failure_reason": None,
                "failure_reason_codes": [],
                "failure_reason_notes": None,
            },
            {
                "question_id": "s4-dev-002",
                "generation_status": "failed",
                "visible_answer": None,
                "review_status": "non_evaluable",
                "human_faithfulness_score": None,
                "human_answer_relevance_score": None,
                "primary_failure_reason": "generation_contract_failure",
                "failure_reason_codes": ["generation_contract_failure"],
                "failure_reason_notes": "No citation.",
            },
        ],
    }
    client = FixedGenerationClient(
        [
            GenerationCompletion(
                content=json.dumps(
                    {
                        "faithfulness_score": 2,
                        "answer_relevance_score": 1,
                        "rationale": "Evidence supports the answer.",
                    }
                ),
                request_id="judge-1",
                model="deepseek-flash",
                finish_reason="completed",
                input_tokens=100,
                cached_input_tokens=20,
                output_tokens=10,
            )
        ]
    )
    config = load_generation_config(
        Path("configs/generation/s7-deepseek-flash-v1.json")
    )
    metadata = {
        "schema_version": 1,
        "run_id": "judge-test",
        "split": "development",
        "judge_prompt_version": "s7-judge-v1",
        "judge_prompt_sha256": judge_prompt_sha256(),
        "judge_output_schema_sha256": judge_output_schema_sha256(),
        "provider": config.provider,
        "protocol": config.protocol,
        "model": config.model,
        "same_model_as_generator": True,
    }
    first = asyncio.run(
        run_judge_evaluation(
            generation_rows,
            human_review,
            tmp_path,
            client,
            config,
            metadata,
        )
    )
    second = asyncio.run(
        run_judge_evaluation(
            generation_rows,
            human_review,
            tmp_path,
            client,
            config,
            metadata,
        )
    )
    assert first == second
    assert first["operations"]["success_count"] == 1
    assert first["operations"]["failed_count"] == 0
    assert first["agreement"]["judge_human_intersection_count"] == 1
    assert first["agreement"]["answer_relevance"]["accuracy"] == 0.0
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert [row["question_id"] for row in rows] == ["s4-dev-001"]
    with pytest.raises(ValueError, match="0, 1, or 2"):
        parse_judge_output(
            '{"faithfulness_score":3,"answer_relevance_score":1,"rationale":"bad"}'
        )


def test_fixed_stratified_sample_and_template_hash() -> None:
    questions = [
        _question(4, "single_fact", True),
        _question(1, "single_fact", True),
        _question(5, "cross_section", True),
        _question(2, "single_fact", True),
        _question(6, "cross_section", True),
        _question(8, "out_of_domain", False),
        _question(3, "single_fact", True),
        _question(7, "out_of_domain", False),
    ]
    sample = select_review_sample(questions, per_stratum=2)
    assert [item.question_id for item in sample] == [
        "s4-dev-005",
        "s4-dev-006",
        "s4-dev-007",
        "s4-dev-008",
        "s4-dev-001",
        "s4-dev-002",
    ]
    results = [
        {
            "question_id": item.question_id,
            "status": "success",
            "answer": f"Visible answer for {item.question_id}",
        }
        for item in sample
    ]
    first = create_review_template(sample, results)
    second = create_review_template(list(reversed(sample)), list(reversed(results)))
    assert first == second
    assert len(first["template_sha256"]) == 64
    assert all(row["human_faithfulness_score"] is None for row in first["rows"])
    assert all(row["generation_status"] == "success" for row in first["rows"])
    assert all(row["visible_answer"] for row in first["rows"])
    assert all(row["review_status"] is None for row in first["rows"])


def test_review_template_preserves_failed_fixed_sample_without_replacement() -> None:
    questions = [_question(1, "single_fact", True), _question(2, "single_fact", True)]
    template = create_review_template(
        questions,
        [
            {"question_id": "s4-dev-001", "status": "success", "answer": "Visible."},
            {
                "question_id": "s4-dev-002",
                "status": "failed",
                "answer": None,
                "error_type": "GenerationValidationError",
            },
        ],
    )
    assert [row["question_id"] for row in template["rows"]] == [
        "s4-dev-001",
        "s4-dev-002",
    ]
    failed = template["rows"][1]
    assert failed["generation_status"] == "failed"
    assert failed["visible_answer"] is None
    assert failed["review_status"] == "non_evaluable"
    assert failed["primary_failure_reason"] == "generation_contract_failure"
    assert failed["failure_reason_codes"] == ["generation_contract_failure"]


def test_human_review_contract_and_denominators() -> None:
    rows = [
        {
            "question_id": "q1",
            "generation_status": "success",
            "visible_answer": "Supported answer.",
            "review_status": "evaluable",
            "human_faithfulness_score": 2,
            "human_answer_relevance_score": 1,
            "primary_failure_reason": None,
            "failure_reason_codes": [],
            "failure_reason_notes": None,
        },
        {
            "question_id": "q2",
            "generation_status": "failed",
            "visible_answer": None,
            "review_status": "non_evaluable",
            "human_faithfulness_score": None,
            "human_answer_relevance_score": None,
            "primary_failure_reason": "retrieval_miss",
            "failure_reason_codes": [
                "retrieval_miss",
                "generation_contract_failure",
            ],
            "failure_reason_notes": "Dense Top-10 did not contain a gold chunk.",
        },
    ]
    summary = summarize_human_review(
        rows,
        fixed_sample_count=2,
        full_split_count=60,
        full_split_failure_count=11,
    )
    assert summary == {
        "fixed_sample_count": 2,
        "n_evaluable": 1,
        "n_non_evaluable": 1,
        "fixed_sample_generation_failure_rate": 0.5,
        "full_split_count": 60,
        "full_split_generation_failure_count": 11,
        "full_split_generation_failure_rate": 11 / 60,
        "failure_reason_question_counts": {
            "generation_contract_failure": 1,
            "retrieval_miss": 1,
        },
        "failure_reason_counts_are_multilabel": True,
        "human_mean_faithfulness": 2.0,
        "human_mean_answer_relevance": 1.0,
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"visible_answer": None}, "visible answer"),
        ({"human_faithfulness_score": None}, "0, 1, or 2"),
        ({"failure_reason_codes": ["retrieval_miss"]}, "must not have failure reasons"),
    ],
)
def test_evaluable_review_contract_is_strict(updates, message) -> None:
    row = {
        "question_id": "q1",
        "generation_status": "success",
        "visible_answer": "Answer.",
        "review_status": "evaluable",
        "human_faithfulness_score": 2,
        "human_answer_relevance_score": 2,
        "primary_failure_reason": None,
        "failure_reason_codes": [],
        "failure_reason_notes": None,
    }
    row.update(updates)
    with pytest.raises(HumanReviewError, match=message):
        summarize_human_review([row], fixed_sample_count=1, full_split_count=60)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"human_faithfulness_score": 0}, "must be null"),
        ({"failure_reason_codes": []}, "at least one failure reason"),
        ({"primary_failure_reason": "upstream_failure"}, "must belong"),
        (
            {
                "primary_failure_reason": "unknown",
                "failure_reason_codes": ["unknown"],
                "failure_reason_notes": None,
            },
            "unknown requires",
        ),
        (
            {
                "primary_failure_reason": "unknown",
                "failure_reason_codes": ["unknown", "retrieval_miss"],
                "failure_reason_notes": "unclear",
            },
            "unknown must be exclusive",
        ),
        (
            {
                "primary_failure_reason": "dataset_question_ambiguity",
                "failure_reason_codes": ["dataset_question_ambiguity"],
                "failure_reason_notes": "The object is missing.",
                "failure_reason_confirmed_by": None,
            },
            "ambiguity requires human confirmation",
        ),
    ],
)
def test_non_evaluable_review_contract_is_strict(updates, message) -> None:
    row = {
        "question_id": "q1",
        "generation_status": "failed",
        "visible_answer": None,
        "review_status": "non_evaluable",
        "human_faithfulness_score": None,
        "human_answer_relevance_score": None,
        "primary_failure_reason": "generation_contract_failure",
        "failure_reason_codes": ["generation_contract_failure"],
        "failure_reason_notes": "No citations were returned.",
        "failure_reason_confirmed_by": None,
    }
    row.update(updates)
    with pytest.raises(HumanReviewError, match=message):
        summarize_human_review([row], fixed_sample_count=1, full_split_count=60)


def test_sample_rejects_test_questions() -> None:
    payload = _question(1, "single_fact", True).to_dict()
    payload["question_id"] = "s4-test-001"
    payload["split"] = "test"
    with pytest.raises(TestSplitLockedError, match="S12"):
        select_review_sample([EvaluationQuestion.from_dict(payload)])


def test_compare_requires_complete_unique_human_rows() -> None:
    judges = {
        "q1": parse_judge_output(
            '{"faithfulness_score":2,"answer_relevance_score":1,"rationale":"a"}'
        ),
    }
    incomplete = [
        {
            "question_id": "q1",
            "generation_status": "success",
            "visible_answer": "Answer 1.",
            "review_status": "evaluable",
            "human_faithfulness_score": 2,
            "human_answer_relevance_score": 1,
            "primary_failure_reason": None,
            "failure_reason_codes": [],
            "failure_reason_notes": None,
        },
        {
            "question_id": "q2",
            "generation_status": "success",
            "visible_answer": "Answer 2.",
            "review_status": "evaluable",
            "human_faithfulness_score": 0,
            "human_answer_relevance_score": 2,
            "primary_failure_reason": None,
            "failure_reason_codes": [],
            "failure_reason_notes": None,
        },
    ]
    summary = compare_judge_to_human(judges, incomplete, require_complete=False)
    assert summary["human_validation_status"] == "complete"
    assert summary["judge_human_intersection_count"] == 1
    assert summary["judge_missing_count"] == 1
    with pytest.raises(HumanReviewError, match="missing"):
        compare_judge_to_human(judges, incomplete)
    unavailable = compare_judge_to_human({}, incomplete, require_complete=False)
    assert unavailable["judge_metrics"] == "unavailable"
    assert unavailable["judge_human_intersection_count"] == 0
    assert unavailable["judge_missing_count"] == 2


def test_agreement_confusion_weighted_score_and_kappa() -> None:
    judges = {
        "q1": parse_judge_output(
            '{"faithfulness_score":2,"answer_relevance_score":1,"rationale":"a"}'
        ),
        "q2": parse_judge_output(
            '{"faithfulness_score":0,"answer_relevance_score":2,"rationale":"b"}'
        ),
    }
    humans = [
        {
            "question_id": "q1",
            "generation_status": "success",
            "visible_answer": "Answer 1.",
            "review_status": "evaluable",
            "human_faithfulness_score": 2,
            "human_answer_relevance_score": 2,
            "primary_failure_reason": None,
            "failure_reason_codes": [],
            "failure_reason_notes": None,
        },
        {
            "question_id": "q2",
            "generation_status": "success",
            "visible_answer": "Answer 2.",
            "review_status": "evaluable",
            "human_faithfulness_score": 1,
            "human_answer_relevance_score": 2,
            "primary_failure_reason": None,
            "failure_reason_codes": [],
            "failure_reason_notes": None,
        },
    ]
    result = compare_judge_to_human(judges, humans)
    assert result["human_validation_status"] == "complete"
    assert result["faithfulness"]["accuracy"] == 0.5
    assert result["faithfulness"]["weighted_score_agreement"] == 0.75
    assert result["faithfulness"]["confusion_matrix"][2][2] == 1
    assert result["disagreements"] == [
        {
            "question_id": "q1",
            "dimension": "answer_relevance",
            "judge_score": 1,
            "human_score": 2,
        },
        {
            "question_id": "q2",
            "dimension": "faithfulness",
            "judge_score": 0,
            "human_score": 1,
        },
    ]
    constant = compare_judge_to_human(
        {"q": parse_judge_output(json.dumps({"faithfulness_score": 2, "answer_relevance_score": 2, "rationale": "x"}))},
        [
            {
                "question_id": "q",
                "generation_status": "success",
                "visible_answer": "Answer.",
                "review_status": "evaluable",
                "human_faithfulness_score": 2,
                "human_answer_relevance_score": 2,
                "primary_failure_reason": None,
                "failure_reason_codes": [],
                "failure_reason_notes": None,
            }
        ],
    )
    assert constant["faithfulness"]["cohens_kappa"] == 1.0

