"""Deterministic Server-Sent Events framing for S9 queries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Protocol


class DisconnectProbe(Protocol):
    async def is_disconnected(self) -> bool: ...


def _frame(event: str, payload: object) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


async def stream_query(
    request: DisconnectProbe,
    question: str,
    requested_version: str | None,
    service: Any,
    *,
    request_id: str,
    poll_seconds: float = 0.05,
) -> AsyncIterator[str]:
    yield _frame("started", {"request_id": request_id})
    task = asyncio.create_task(
        service.query(question, requested_version, request_id=request_id)
    )
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=poll_seconds)
            if done:
                break
            if await request.is_disconnected():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                return
            await asyncio.sleep(0)
        outcome = await task
        if outcome.http_status >= 500:
            yield _frame(
                "error",
                {
                    "http_status": outcome.http_status,
                    "reason_code": "UPSTREAM_ERROR",
                    "request_id": request_id,
                },
            )
            yield _frame("done", {"status": "failed"})
            return
        yield _frame("result", outcome.response.model_dump(mode="json"))
        yield _frame("done", {"status": "complete"})
    except asyncio.CancelledError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    except Exception:  # noqa: BLE001 - fail closed at the streaming boundary
        yield _frame(
            "error",
            {
                "http_status": 503,
                "reason_code": "UPSTREAM_ERROR",
                "request_id": request_id,
            },
        )
        yield _frame("done", {"status": "failed"})
