"""Allow-list-only structured HTTP logging."""

from __future__ import annotations

import json
import logging


def log_request(
    logger: logging.Logger,
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    elapsed_ms: float,
) -> None:
    payload = {
        "elapsed_ms": round(elapsed_ms, 3),
        "event": "http_request",
        "method": method,
        "path": path,
        "request_id": request_id,
        "status_code": status_code,
    }
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
