from __future__ import annotations

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from docguard.api.app import create_app
from docguard.api.config import ApiConfig
from docguard.api.dependencies import AppServices
from docguard.api.models import DecisionResponse, QueryResponse
from docguard.service.artifacts import (
    CorpusArtifact,
    EvidenceArtifact,
    EvidenceNotFoundError,
)
from docguard.service.evaluations import EvaluationJob, EvaluationJobNotFound
from docguard.service.query import QueryOutcome


def _config() -> ApiConfig:
    return ApiConfig(
        1, "s9-api-v1", "/api/v1", ("http://localhost:5173",), 1.0, 2000,
        ("b0", "b1", "m1", "m2", "b2", "m4"),
        ("development", "validation"), "locked_until_s12"
    )


class _Query:
    mode = "answer"

    async def query(self, question: str, requested_version: str | None, *, request_id: str) -> QueryOutcome:
        action = "error" if self.mode in {"error", "timeout"} else self.mode
        reason = "UPSTREAM_ERROR" if action == "error" else (
            "NO_RELEVANT_EVIDENCE" if action == "refuse" else None
        )
        response = QueryResponse(
            request_id=request_id,
            question=question,
            requested_version=requested_version,
            route="m4",
            decision=DecisionResponse(
                action=action,
                reason_code=reason,
                confidence=1.0,
                triggered_rule="fixture",
            ),
            answer="supported answer" if action == "answer" else None,
            citations=[],
            evidence=[],
            latency_ms={"end_to_end": 1.0},
            model_info=None,
        )
        status = 504 if self.mode == "timeout" else (503 if action == "error" else 200)
        return QueryOutcome(response, status)

    async def close(self) -> None:
        return None


class _Artifacts:
    async def rebuild(self) -> str:
        return "rebuild-1"

    def list_corpora(self) -> tuple[CorpusArtifact, ...]:
        return (CorpusArtifact("18.6", 1, 2, "ready", "a" * 64, "2026-10-20T00:00:00Z"),)

    def get_evidence(self, evidence_id: str) -> EvidenceArtifact:
        if evidence_id == "missing":
            raise EvidenceNotFoundError("not found")
        return EvidenceArtifact(
            evidence_id, "18.6", "admin", ("Reference", "VACUUM"),
            "https://example/18", "vacuum", "VACUUM text", "b" * 64, "s2-chunk-v1"
        )


@dataclass
class _Evaluations:
    job: EvaluationJob = field(
        default_factory=lambda: EvaluationJob(
            "run-1",
            "m4",
            "validation",
            "queued",
            "2026-10-20T00:00:00Z",
            "2026-10-20T00:00:00Z",
        )
    )

    async def create_job(self, *, method: str, split: str) -> EvaluationJob:
        self.job = EvaluationJob(
            "run-1", method, split, "queued", "2026-10-20T00:00:00Z", "2026-10-20T00:00:00Z"
        )
        return self.job

    def list_jobs(self) -> tuple[EvaluationJob, ...]:
        return (self.job,)

    def get_job(self, run_id: str) -> EvaluationJob:
        if run_id != self.job.run_id:
            raise EvaluationJobNotFound("not found")
        return self.job

    async def close(self) -> None:
        return None


def _client(*, healthy: bool = True) -> tuple[TestClient, _Query]:
    query = _Query()
    services = AppServices(
        query=query,
        artifacts=_Artifacts(),
        evaluations=_Evaluations(),
        dependency_probe=lambda: {"corpus": (healthy, None if healthy else "missing")},
    )
    return TestClient(create_app(_config(), services)), query


def test_openapi_contains_all_non_streaming_routes() -> None:
    client, _ = _client()
    paths = client.get("/openapi.json").json()["paths"]

    assert {
        "/api/v1/health", "/api/v1/corpora", "/api/v1/corpora/rebuild",
        "/api/v1/query", "/api/v1/evidence/{evidence_id}",
        "/api/v1/evaluations", "/api/v1/evaluations/{run_id}",
    }.issubset(paths)


def test_health_distinguishes_dependency_unavailability() -> None:
    client, _ = _client(healthy=False)
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["dependencies"][0]["status"] == "unavailable"


def test_corpora_rebuild_and_evidence_routes() -> None:
    client, _ = _client()

    assert client.get("/api/v1/corpora").json()["corpora"][0]["version"] == "18.6"
    rebuild = client.post("/api/v1/corpora/rebuild")
    assert rebuild.status_code == 202
    assert rebuild.json()["job_id"] == "rebuild-1"
    assert client.get("/api/v1/evidence/chk_18").json()["chunk_id"] == "chk_18"
    assert client.get("/api/v1/evidence/missing").status_code == 404


def test_query_answer_refusal_and_upstream_error() -> None:
    client, query = _client()

    answer = client.post("/api/v1/query", json={"question": "VACUUM?"})
    assert answer.status_code == 200
    assert answer.json()["decision"]["action"] == "answer"
    assert answer.headers["x-request-id"] == answer.json()["request_id"]

    query.mode = "refuse"
    refusal = client.post("/api/v1/query", json={"question": "unknown"})
    assert refusal.status_code == 200
    assert refusal.json()["decision"]["action"] == "refuse"

    query.mode = "error"
    error = client.post("/api/v1/query", json={"question": "VACUUM?"})
    assert error.status_code == 503
    assert error.json()["decision"]["reason_code"] == "UPSTREAM_ERROR"


def test_query_validation_error_is_422() -> None:
    client, _ = _client()

    assert client.post("/api/v1/query", json={"question": "", "unexpected": 1}).status_code == 422


def test_evaluation_creation_listing_and_lookup() -> None:
    client, _ = _client()
    created = client.post(
        "/api/v1/evaluations", json={"method": "m4", "split": "validation"}
    )

    assert created.status_code == 202
    assert created.json()["run_id"] == "run-1"
    assert client.get("/api/v1/evaluations").json()["evaluations"][0]["run_id"] == "run-1"
    assert client.get("/api/v1/evaluations/run-1").status_code == 200
    assert client.get("/api/v1/evaluations/missing").status_code == 404


def test_cors_allows_only_configured_vite_origin() -> None:
    client, _ = _client()
    allowed = client.options(
        "/api/v1/query",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    denied = client.options(
        "/api/v1/query",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in denied.headers
