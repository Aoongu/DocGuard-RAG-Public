from __future__ import annotations

import asyncio

import pytest
from test_s9_api import _client

from docguard.api.sse import stream_query


def test_sse_success_has_result_and_terminal_done() -> None:
    client, _ = _client()

    with client.stream(
        "POST", "/api/v1/query/stream", json={"question": "VACUUM?"}
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: started" in body
    assert "event: result" in body
    assert "event: done" in body
    assert '"status":"complete"' in body


def test_sse_upstream_error_is_explicit() -> None:
    client, query = _client()
    query.mode = "error"

    body = client.post(
        "/api/v1/query/stream", json={"question": "VACUUM?"}
    ).text

    assert "event: error" in body
    assert '"reason_code":"UPSTREAM_ERROR"' in body
    assert '"status":"failed"' in body


def test_sse_timeout_preserves_gateway_timeout_status() -> None:
    client, query = _client()
    query.mode = "timeout"

    body = client.post(
        "/api/v1/query/stream", json={"question": "VACUUM?"}
    ).text

    assert "event: error" in body
    assert '"http_status":504' in body
    assert '"status":"failed"' in body


class _DisconnectingRequest:
    def __init__(self) -> None:
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls >= 2


class _SlowQuery:
    cancelled = False

    async def query(self, *args, **kwargs):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_sse_disconnect_cancels_query_task() -> None:
    request = _DisconnectingRequest()
    service = _SlowQuery()
    events = []

    async for event in stream_query(
        request, "VACUUM?", None, service, request_id="req-1", poll_seconds=0
    ):
        events.append(event)

    assert len(events) == 1
    assert "event: started" in events[0]
    assert service.cancelled is True


class _ExplodingQuery:
    async def query(self, *args, **kwargs):
        raise RuntimeError("secret upstream details")


@pytest.mark.asyncio
async def test_sse_midstream_exception_is_redacted() -> None:
    request = _DisconnectingRequest()
    request.calls = -100
    events = [
        event
        async for event in stream_query(
            request, "VACUUM?", None, _ExplodingQuery(), request_id="req-2", poll_seconds=0
        )
    ]

    body = "".join(events)
    assert "event: error" in body
    assert "UPSTREAM_ERROR" in body
    assert "secret upstream details" not in body
