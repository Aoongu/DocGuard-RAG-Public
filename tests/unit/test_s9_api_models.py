from __future__ import annotations

import pytest
from pydantic import ValidationError

from docguard.api.models import (
    DecisionResponse,
    EvaluationCreateRequest,
    EvaluationJobResponse,
    QueryRequest,
    QueryResponse,
)


def test_query_request_normalizes_question() -> None:
    request = QueryRequest(question="  How does VACUUM work?  ", requested_version="18.6")

    assert request.question == "How does VACUUM work?"


@pytest.mark.parametrize("version", ["19", "18", "latest", 18])
def test_query_request_rejects_unknown_version(version: object) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="How does VACUUM work?", requested_version=version)


def test_query_request_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        QueryRequest(question="VACUUM?", secret="value")


def test_query_response_requires_answer_for_answer_action() -> None:
    with pytest.raises(ValidationError, match="answer action"):
        QueryResponse(
            request_id="req-1",
            question="VACUUM?",
            requested_version=None,
            route="m4",
            decision=DecisionResponse(
                action="answer", reason_code=None, confidence=0.8, triggered_rule="ok"
            ),
            answer=None,
            citations=[],
            evidence=[],
            latency_ms={"end_to_end": 1.0},
            model_info=None,
        )


def test_error_response_requires_upstream_reason() -> None:
    with pytest.raises(ValidationError, match="UPSTREAM_ERROR"):
        DecisionResponse(
            action="error",
            reason_code="NO_RELEVANT_EVIDENCE",
            confidence=1.0,
            triggered_rule="upstream",
        )


def test_evaluation_create_rejects_test_split() -> None:
    with pytest.raises(ValidationError):
        EvaluationCreateRequest(method="m4", split="test")


def test_failed_job_requires_failure_information() -> None:
    with pytest.raises(ValidationError, match="failed job"):
        EvaluationJobResponse(
            run_id="run-1",
            method="m4",
            split="validation",
            status="failed",
            created_at="2026-10-20T00:00:00Z",
            updated_at="2026-10-20T00:00:01Z",
        )
