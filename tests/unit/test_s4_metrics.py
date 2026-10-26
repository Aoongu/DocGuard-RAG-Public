import math

import pytest

from docguard.evaluation.metrics import (
    aggregate_retrieval_metrics,
    retrieval_metrics,
    summarize_operations,
)


def test_retrieval_metrics_match_hand_calculation() -> None:
    metrics = retrieval_metrics(
        retrieved_chunk_ids=("x", "r1", "y", "r2"),
        relevant_chunk_ids=("r1", "r2"),
        cutoffs=(1, 5, 10),
        mrr_cutoff=10,
        ndcg_cutoff=10,
    )

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr_at_10"] == 0.5
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    expected_idcg = 1 + 1 / math.log2(3)
    assert metrics["ndcg_at_10"] == pytest.approx(expected_dcg / expected_idcg)


def test_metrics_reject_empty_relevance_and_duplicate_retrieval_ids() -> None:
    with pytest.raises(ValueError, match="relevant"):
        retrieval_metrics(("a",), (), (1,), 10, 10)
    with pytest.raises(ValueError, match="duplicate"):
        retrieval_metrics(("a", "a"), ("a",), (1,), 10, 10)


def test_aggregate_and_operational_summary_are_deterministic() -> None:
    rows = [
        {
            "question_id": "q1",
            "status": "success",
            "answerable": True,
            "metrics": {"recall_at_1": 1.0, "mrr_at_10": 1.0},
            "latency_ms": {"retrieval": 10.0, "end_to_end": 12.0},
            "usage": {"input_tokens": 0, "output_tokens": 0, "api_cost_usd": 0.0},
        },
        {
            "question_id": "q2",
            "status": "success",
            "answerable": True,
            "metrics": {"recall_at_1": 0.0, "mrr_at_10": 0.5},
            "latency_ms": {"retrieval": 30.0, "end_to_end": 35.0},
            "usage": {"input_tokens": 3, "output_tokens": 2, "api_cost_usd": 0.01},
        },
        {
            "question_id": "q3",
            "status": "failed",
            "answerable": False,
            "metrics": {},
            "latency_ms": {},
            "usage": {"input_tokens": 0, "output_tokens": 0, "api_cost_usd": 0.0},
        },
    ]

    assert aggregate_retrieval_metrics(rows) == {
        "evaluated_answerable_count": 2,
        "mrr_at_10": 0.75,
        "recall_at_1": 0.5,
    }
    summary = summarize_operations(rows)
    assert summary["latency_ms"]["retrieval"] == {"p50": 20.0, "p95": 29.0}
    assert summary["usage"] == {
        "input_tokens": 3,
        "output_tokens": 2,
        "api_cost_usd": 0.01,
    }
    assert summary["failed_count"] == 1

