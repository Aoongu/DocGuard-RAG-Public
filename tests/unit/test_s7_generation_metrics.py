import pytest

from docguard.evaluation.generation_metrics import (
    aggregate_generation_metrics,
    citation_metrics,
)


def test_citation_metrics_exact_partial_and_empty() -> None:
    assert citation_metrics(("a", "b"), ("a", "b"), ("a", "b", "c")) == {
        "citation_precision": 1.0,
        "citation_recall": 1.0,
    }
    assert citation_metrics(("a", "c"), ("a", "b"), ("a", "b", "c")) == {
        "citation_precision": 0.5,
        "citation_recall": 0.5,
    }
    assert citation_metrics((), ("a",), ("a",)) == {
        "citation_precision": 0.0,
        "citation_recall": 0.0,
    }


@pytest.mark.parametrize(
    ("cited", "gold", "known", "message"),
    [
        (("a", "a"), ("a",), ("a",), "duplicate cited"),
        (("a",), ("a", "a"), ("a",), "duplicate relevant"),
        (("a",), ("missing",), ("a",), "unknown gold"),
        (("missing",), ("a",), ("a",), "unknown cited"),
    ],
)
def test_citation_metrics_reject_invalid_sets(cited, gold, known, message) -> None:
    with pytest.raises(ValueError, match=message):
        citation_metrics(cited, gold, known)


def test_aggregate_separates_answerable_and_boundary_rows() -> None:
    rows = [
        {
            "status": "success",
            "answerable": True,
            "answer": "answer",
            "citations": [{"chunk_id": "a"}],
            "metrics": {"citation_precision": 1.0, "citation_recall": 0.5},
        },
        {
            "status": "success",
            "answerable": False,
            "answer": "conservative",
            "citations": [],
            "metrics": {},
        },
        {
            "status": "success",
            "answerable": False,
            "answer": "",
            "citations": [{"chunk_id": "x"}],
            "metrics": {},
        },
        {"status": "failed", "answerable": True, "metrics": {}},
    ]

    summary = aggregate_generation_metrics(rows)

    assert summary["answerable"] == {
        "evaluated_count": 1,
        "citation_precision": 1.0,
        "citation_recall": 0.5,
    }
    assert summary["boundary"] == {
        "evaluated_count": 2,
        "answer_rate": 0.5,
        "citation_rate": 0.5,
    }

