"""HTTP adapters for persistent document discovery and import tasks."""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from docguard.discovery.models import Approval, CreateTask, KnowledgeQuery, Reply
from docguard.discovery.store import Conflict, NotFound
from docguard.fs_safety import FileSafetyError


def _translate(error):
    if isinstance(error, NotFound):
        return HTTPException(404, detail={"type": "not_found"})
    if isinstance(error, Conflict):
        return HTTPException(409, detail={"type": "state_conflict"})
    if isinstance(error, (OSError, RuntimeError, FileSafetyError)):
        return HTTPException(503, detail={"type": "local_artifact_unavailable"})
    return HTTPException(422, detail={"type": "invalid_request"})


def _call(function, *args):
    try:
        return function(*args)
    except (
        NotFound,
        Conflict,
        OSError,
        RuntimeError,
        ValueError,
        FileSafetyError,
    ) as error:
        raise _translate(error) from None


async def _await(coroutine):
    try:
        return await coroutine
    except (
        NotFound,
        Conflict,
        OSError,
        RuntimeError,
        ValueError,
        FileSafetyError,
    ) as error:
        raise _translate(error) from None


def discovery_router(service):
    router = APIRouter()

    def require_service():
        if service is None:
            raise HTTPException(503, detail={"type": "discovery_unavailable"})
        return service

    @router.post("/discovery/tasks", status_code=202)
    async def create_task(payload: CreateTask):
        worker = require_service()
        return await _await(
            worker.create(
                payload.goal,
                payload.product,
                payload.version,
                payload.urls,
                official_only=payload.official_only,
                document_type=payload.document_type,
            )
        )

    @router.get("/discovery/tasks/{task_id}")
    def get_task(task_id: str):
        return _call(require_service().get_task, task_id)

    @router.post("/discovery/tasks/{task_id}/reply", status_code=202)
    async def reply(task_id: str, payload: Reply):
        return await _await(
            require_service().reply(
                task_id,
                payload.urls,
                message=payload.message,
                version=payload.version,
                product=payload.product,
                allow_unversioned=payload.allow_unversioned,
            )
        )

    @router.get("/discovery/capabilities")
    def capabilities():
        return require_service().capabilities()

    @router.post("/discovery/tasks/{task_id}/approve", status_code=202)
    async def approve(task_id: str, payload: Approval):
        return await _await(
            require_service().approve(
                task_id,
                payload.candidate_ids,
                payload.revision,
                payload.knowledge_base_id,
                payload.idempotency_key,
                import_scope=payload.import_scope,
            )
        )

    @router.post("/discovery/tasks/{task_id}/cancel")
    async def cancel(task_id: str):
        return await _await(require_service().cancel(task_id))

    @router.post("/imports/{job_id}/retry", status_code=202)
    async def retry_import(job_id: str):
        return await _await(require_service().retry_import(job_id))

    @router.get("/imports/{job_id}")
    def get_import(job_id: str):
        return _call(require_service().get_job, job_id)

    @router.get("/discovery/tasks/{task_id}/events")
    async def events(task_id: str, request: Request):
        worker = require_service()
        _call(worker.get_task, task_id)
        try:
            after = int(request.headers.get("last-event-id", "0"))
            if after < 0:
                raise ValueError("Negative event ID")
        except ValueError:
            raise HTTPException(422, detail={"type": "invalid_event_cursor"}) from None

        async def stream():
            cursor = after
            while not await request.is_disconnected():
                batch = worker.events(task_id, cursor)
                for event in batch:
                    cursor = event["event_id"]
                    yield f"id: {cursor}\nevent: {event['kind']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                if len(batch) == 500:
                    continue
                if worker.get_task(task_id)["status"] not in (
                    "discovering",
                    "importing",
                ):
                    return
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/knowledge-bases")
    def list_bases():
        return {"knowledge_bases": _call(require_service().list_knowledge_bases)}

    @router.post("/knowledge-bases/{kb_id}/query")
    async def query(kb_id: str, payload: KnowledgeQuery):
        return await _await(
            require_service().query(kb_id, payload.question, generate=payload.generate)
        )

    @router.delete("/knowledge-bases/{kb_id}")
    def delete_base(kb_id: str):
        return _call(require_service().delete_knowledge_base, kb_id)

    return router
