from __future__ import annotations

import json
from pathlib import Path

import pytest

from docguard.evaluation.final_report import (
    build_human_review_packet,
    build_method_summary,
    build_paired_comparison,
    file_sha256,
    human_review_summary,
    percentile,
    resolve_human_review_packet,
)


def _row(
    question_id: str,
    *,
    status: str = "success",
    answerable: bool = True,
    action: str = "answer",
    expected_action: str = "answer",
    metric: float = 1.0,
    cost: float = 0.25,
    error_type: str | None = None,
    reason_code: str | None = None,
) -> dict[str, object]:
    answer = "supported answer" if status == "success" and action == "answer" else None
    return {
        "question_id": question_id,
        "question": f"question {question_id}",
        "status": status,
        "answerable": answerable,
        "action": action,
        "expected_action": expected_action,
        "reason_code": reason_code,
        "answer": answer,
        "citations": [{"chunk_id": "chunk-1"}] if answer else [],
        "evidence": [{"chunk_id": "chunk-1", "text": "evidence"}],
        "metrics": {
            "recall_at_10": metric,
            "mrr_at_10": metric,
            "ndcg_at_10": metric,
            "citation_precision": metric,
            "citation_recall": metric,
        },
        "latency_ms": {"end_to_end": metric * 100.0},
        "usage": {"input_tokens": 1, "output_tokens": 1, "api_cost_usd": cost},
        "error_type": error_type,
        "error_message": "failure" if error_type else None,
    }


def test_percentile_matches_hand_calculated_p50_and_p95() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.50) == 25.0
    assert percentile(values, 0.95) == pytest.approx(38.5)


def test_method_summary_uses_fixed_denominator_and_separates_judge() -> None:
    rows = [
        _row("q1", metric=1.0, cost=0.1),
        _row("q2", metric=0.0, cost=0.2),
        _row(
            "q3",
            status="failed",
            action="error",
            expected_action="answer",
            metric=0.0,
            cost=0.3,
            error_type="CitationValidationFailure",
        ),
    ]
    summary = build_method_summary("m4", rows, fixed_total=3)

    assert summary["fixed_total"] == 3
    assert summary["n_evaluable"] == 2
    assert summary["n_non_evaluable"] == 1
    assert summary["n_not_applicable"] == 0
    assert summary["generation_failure_rate"] == pytest.approx(1 / 3)
    assert summary["api_cost_usd"] == pytest.approx(0.6)
    assert summary["latency_ms"]["end_to_end"]["p50"] == 0.0
    assert summary["latency_ms"]["end_to_end"]["p95"] == pytest.approx(90.0)
    assert summary["failure_reason_counts"] == {"generation_contract_failure": 1}
    assert summary["judge"] == {"status": "not_run", "failed_count": 0}


def test_hand_calculated_retrieval_citation_and_refusal_metrics() -> None:
    rows = [
        _row("q1", metric=1.0),
        _row("q2", metric=0.5),
        _row(
            "q3",
            answerable=False,
            action="answer",
            expected_action="refuse",
            metric=0.0,
        ),
        _row(
            "q4",
            answerable=False,
            action="refuse",
            expected_action="refuse",
            metric=0.0,
        ),
    ]
    summary = build_method_summary("m4", rows, fixed_total=4)

    assert summary["retrieval"]["recall_at_10"] == pytest.approx(0.75)
    assert summary["retrieval"]["mrr_at_10"] == pytest.approx(0.75)
    assert summary["retrieval"]["ndcg_at_10"] == pytest.approx(0.75)
    assert summary["generation"]["citation_precision"] == pytest.approx(0.75)
    assert summary["generation"]["citation_recall"] == pytest.approx(0.75)
    assert summary["refusal"]["false_accept_count"] == 1
    assert summary["refusal"]["false_reject_count"] == 0
    assert summary["refusal"]["refusal_precision"] == 1.0
    assert summary["refusal"]["refusal_recall"] == 0.5
    assert summary["refusal"]["refusal_f1"] == pytest.approx(2 / 3)


def test_paired_bootstrap_is_deterministic_and_uses_intersection() -> None:
    baseline = [_row("q1", metric=0.0), _row("q2", metric=1.0), _row("q3", metric=0.0)]
    candidate = [_row("q1", metric=1.0), _row("q2", metric=1.0), _row("q4", metric=1.0)]

    left = build_paired_comparison(
        baseline,
        candidate,
        metric_name="recall_at_10",
        samples=10_000,
        seed=20261014,
        confidence_level=0.95,
    )
    right = build_paired_comparison(
        baseline,
        candidate,
        metric_name="recall_at_10",
        samples=10_000,
        seed=20261014,
        confidence_level=0.95,
    )

    assert left == right
    assert left["paired_question_count"] == 2
    assert left["mean_difference"] == 0.5
    assert left["ci_lower"] == 0.0
    assert left["ci_upper"] == 1.0


def test_review_packet_is_deterministic_and_keeps_human_scores_pending() -> None:
    rows = [
        _row("q5", status="failed", action="error", error_type="UpstreamAPIError"),
        _row("q2", answerable=False, action="answer", expected_action="refuse"),
        _row("q4", action="refuse", expected_action="answer"),
        _row(
            "q3",
            answerable=False,
            action="clarify",
            expected_action="clarify",
            reason_code="VERSION_CONFLICT",
        ),
        _row("q1"),
    ]

    packet = build_human_review_packet("m4", rows, per_category=1)
    assert [item["sample_category"] for item in packet["samples"]] == [
        "success",
        "false_accept",
        "false_reject",
        "version_conflict",
        "generation_failure",
    ]
    assert [item["question_id"] for item in packet["samples"]] == [
        "q1",
        "q2",
        "q4",
        "q3",
        "q5",
    ]
    for item in packet["samples"]:
        assert item["review_status"] == "pending"
        assert item["faithfulness"] is None
        assert item["answer_relevance"] is None


def test_source_hash_is_written_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "results.jsonl"
    source.write_text(json.dumps({"question_id": "q1"}) + "\n", encoding="utf-8")
    before = source.read_bytes()

    digest = file_sha256(source)

    assert len(digest) == 64
    assert source.read_bytes() == before


def _completed_review_packet() -> dict[str, object]:
    packet = build_human_review_packet(
        "m4",
        [
            _row("q1"),
            _row(
                "q2",
                status="failed",
                action="error",
                expected_action="error",
                error_type="CitationValidationFailure",
            ),
        ],
        per_category=1,
    )
    packet["human_validation_status"] = "complete"
    packet["samples"][0].update(
        {
            "review_status": "evaluable",
            "faithfulness": 2,
            "answer_relevance": 1,
            "reviewer": "project_owner",
        }
    )
    packet["samples"][1].update(
        {
            "review_status": "non_evaluable",
            "faithfulness": None,
            "answer_relevance": None,
            "reviewer": "project_owner",
            "failure_reason_confirmed_by": "project_owner",
        }
    )
    return packet


def test_completed_human_review_is_preserved_and_summarized() -> None:
    template = build_human_review_packet(
        "m4",
        [
            _row("q1"),
            _row(
                "q2",
                status="failed",
                action="error",
                expected_action="error",
                error_type="CitationValidationFailure",
            ),
        ],
        per_category=1,
    )
    completed = _completed_review_packet()

    resolved = resolve_human_review_packet(completed, template)
    summary = human_review_summary(resolved)

    assert resolved is completed
    assert summary == {
        "status": "complete",
        "reviewer": "project_owner",
        "reviewed_count": 2,
        "n_evaluable": 1,
        "n_non_evaluable": 1,
        "faithfulness_mean": 2.0,
        "answer_relevance_mean": 1.0,
        "failure_reason_counts": {"generation_contract_failure": 1},
        "failure_reason_counts_are_multilabel": True,
    }


def test_completed_human_review_rejects_contract_violations() -> None:
    template = build_human_review_packet(
        "m4",
        [_row("q1")],
        per_category=1,
    )
    completed = build_human_review_packet("m4", [_row("q1")], per_category=1)
    completed["human_validation_status"] = "complete"
    completed["samples"][0].update(
        {
            "review_status": "evaluable",
            "faithfulness": None,
            "answer_relevance": 2,
            "reviewer": "project_owner",
        }
    )

    with pytest.raises(ValueError, match="evaluable scores"):
        resolve_human_review_packet(completed, template)
