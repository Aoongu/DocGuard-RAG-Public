"""FastAPI application factory for DocGuard-RAG S9."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from docguard.api.config import ApiConfig
from docguard.api.dependencies import AppServices
from docguard.api.logging import log_request
from docguard.api.models import (
    CorpusListResponse,
    CorpusResponse,
    DependencyHealth,
    EvaluationCreateRequest,
    EvaluationJobResponse,
    EvaluationListResponse,
    EvidenceResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RebuildResponse,
)
from docguard.api.sse import stream_query
from docguard.service.artifacts import ArtifactUnavailableError, EvidenceNotFoundError
from docguard.service.evaluations import EvaluationJobNotFound


def _request_id(request: Request) -> str:
    return request.state.request_id


def _job(value: object) -> EvaluationJobResponse:
    return EvaluationJobResponse.model_validate(asdict(value))


def _problem(request: Request, status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": _request_id(request),
            "error": {"type": error_type, "message": message},
        },
    )


def create_app(config: ApiConfig, services: AppServices) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await services.evaluations.close()
        close = getattr(services.query, "close", None)
        if callable(close):
            await close()
        if callable(services.shutdown):
            await services.shutdown()

    app = FastAPI(
        title="DocGuard-RAG API",
        version=config.api_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )
    http_logger = logging.getLogger("docguard.api.http")

    @app.middleware("http")
    async def request_identity(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.request_id = f"req-{uuid4().hex}"
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        finally:
            log_request(
                http_logger,
                request_id=request.state.request_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

    @app.exception_handler(EvidenceNotFoundError)
    async def evidence_not_found(request: Request, _: EvidenceNotFoundError) -> JSONResponse:
        return _problem(request, 404, "NOT_FOUND", "evidence was not found")

    @app.exception_handler(EvaluationJobNotFound)
    async def evaluation_not_found(request: Request, _: EvaluationJobNotFound) -> JSONResponse:
        return _problem(request, 404, "NOT_FOUND", "evaluation was not found")

    @app.exception_handler(ArtifactUnavailableError)
    async def artifact_unavailable(request: Request, _: ArtifactUnavailableError) -> JSONResponse:
        return _problem(request, 503, "UPSTREAM_ERROR", "required local artifact is unavailable")

    prefix = config.api_prefix

    @app.get(f"{prefix}/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        observed = services.dependency_probe()
        dependencies = [
            DependencyHealth(
                name=name,
                status="available" if available else "unavailable",
                detail=detail,
            )
            for name, (available, detail) in sorted(observed.items())
        ]
        return HealthResponse(
            request_id=_request_id(request),
            status="ok" if all(item.status == "available" for item in dependencies) else "degraded",
            dependencies=dependencies,
        )

    @app.get(f"{prefix}/corpora", response_model=CorpusListResponse)
    async def corpora(request: Request) -> CorpusListResponse:
        rows = services.artifacts.list_corpora()
        return CorpusListResponse(
            request_id=_request_id(request),
            corpora=[CorpusResponse.model_validate(asdict(row)) for row in rows],
        )

    @app.post(f"{prefix}/corpora/rebuild", response_model=RebuildResponse, status_code=202)
    async def rebuild(request: Request) -> RebuildResponse:
        job_id = await services.artifacts.rebuild()
        return RebuildResponse(request_id=_request_id(request), job_id=job_id)

    @app.post(f"{prefix}/query", response_model=QueryResponse)
    async def query(request: Request, payload: QueryRequest):
        outcome = await services.query.query(
            payload.question,
            payload.requested_version,
            request_id=_request_id(request),
        )
        return JSONResponse(
            status_code=outcome.http_status,
            content=outcome.response.model_dump(mode="json"),
        )

    @app.post(f"{prefix}/query/stream", response_class=StreamingResponse)
    async def query_stream(request: Request, payload: QueryRequest) -> StreamingResponse:
        return StreamingResponse(
            stream_query(
                request,
                payload.question,
                payload.requested_version,
                services.query,
                request_id=_request_id(request),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(f"{prefix}/evidence/{{evidence_id}}", response_model=EvidenceResponse)
    async def evidence(request: Request, evidence_id: str) -> EvidenceResponse:
        item = services.artifacts.get_evidence(evidence_id)
        return EvidenceResponse(request_id=_request_id(request), **asdict(item))

    @app.get(f"{prefix}/evaluations", response_model=EvaluationListResponse)
    async def evaluations(request: Request) -> EvaluationListResponse:
        return EvaluationListResponse(
            request_id=_request_id(request),
            evaluations=[_job(item) for item in services.evaluations.list_jobs()],
        )

    @app.post(f"{prefix}/evaluations", response_model=EvaluationJobResponse, status_code=202)
    async def create_evaluation(payload: EvaluationCreateRequest) -> EvaluationJobResponse:
        return _job(
            await services.evaluations.create_job(method=payload.method, split=payload.split)
        )

    @app.get(f"{prefix}/evaluations/{{run_id}}", response_model=EvaluationJobResponse)
    async def evaluation(run_id: str) -> EvaluationJobResponse:
        return _job(services.evaluations.get_job(run_id))

    return app
