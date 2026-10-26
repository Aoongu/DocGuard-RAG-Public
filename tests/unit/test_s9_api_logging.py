from __future__ import annotations

import json
import logging

from docguard.api.logging import log_request


def test_request_log_contains_only_allow_list(caplog) -> None:
    logger = logging.getLogger("docguard.api.test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_request(
            logger,
            request_id="req-1",
            method="POST",
            path="/api/v1/query",
            status_code=200,
            elapsed_ms=1.25,
        )

    payload = json.loads(caplog.messages[0])
    assert payload == {
        "elapsed_ms": 1.25,
        "event": "http_request",
        "method": "POST",
        "path": "/api/v1/query",
        "request_id": "req-1",
        "status_code": 200,
    }
    assert "authorization" not in caplog.text.lower()
    assert "api_key" not in caplog.text.lower()
