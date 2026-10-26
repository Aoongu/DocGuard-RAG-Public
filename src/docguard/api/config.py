"""Strict loader for the frozen S9 HTTP API contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

_FIELDS = {
    "schema_version",
    "api_version",
    "api_prefix",
    "allowed_origins",
    "query_timeout_seconds",
    "max_question_chars",
    "allowed_evaluation_methods",
    "allowed_evaluation_splits",
    "test_execution_status",
}
_METHODS = ("b0", "b1", "m1", "m2", "b2", "m4")
_SPLITS = ("development", "validation")


@dataclass(frozen=True, slots=True)
class ApiConfig:
    schema_version: int
    api_version: str
    api_prefix: str
    allowed_origins: tuple[str, ...]
    query_timeout_seconds: float
    max_question_chars: int
    allowed_evaluation_methods: tuple[str, ...]
    allowed_evaluation_splits: tuple[str, ...]
    test_execution_status: str


def _origins(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("allowed_origins must be a non-empty list")
    if "*" in value:
        raise ValueError("wildcard CORS origin is forbidden")
    origins: list[str] = []
    for origin in value:
        if not isinstance(origin, str):
            raise TypeError("each origin must be a string")
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("origin must contain only scheme, host, and optional port")
        origins.append(origin)
    if len(set(origins)) != len(origins):
        raise ValueError("allowed_origins must be unique")
    return tuple(origins)


def load_api_config(path: Path) -> ApiConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("API config fields do not match the frozen schema")
    if payload["schema_version"] != 1 or payload["api_version"] != "s9-api-v1":
        raise ValueError("unsupported API config identity")
    if payload["api_prefix"] != "/api/v1":
        raise ValueError("api_prefix must be /api/v1")
    timeout = payload["query_timeout_seconds"]
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("query_timeout_seconds must be a positive finite number")
    max_chars = payload["max_question_chars"]
    if type(max_chars) is not int or not 1 <= max_chars <= 10_000:
        raise ValueError("max_question_chars must be an integer from 1 to 10000")
    methods = payload["allowed_evaluation_methods"]
    if methods != list(_METHODS):
        raise ValueError("allowed_evaluation_methods differ from the frozen contract")
    splits = payload["allowed_evaluation_splits"]
    if splits != list(_SPLITS):
        raise ValueError("Test is locked; only development and validation are allowed")
    if payload["test_execution_status"] != "locked_until_s12":
        raise ValueError("test_execution_status must remain locked_until_s12")
    return ApiConfig(
        schema_version=1,
        api_version="s9-api-v1",
        api_prefix="/api/v1",
        allowed_origins=_origins(payload["allowed_origins"]),
        query_timeout_seconds=float(timeout),
        max_question_chars=max_chars,
        allowed_evaluation_methods=_METHODS,
        allowed_evaluation_splits=_SPLITS,
        test_execution_status="locked_until_s12",
    )
