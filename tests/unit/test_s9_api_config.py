from __future__ import annotations

import json
from pathlib import Path

import pytest

from docguard.api.config import load_api_config


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "api_version": "s9-api-v1",
        "api_prefix": "/api/v1",
        "allowed_origins": ["http://localhost:5173"],
        "query_timeout_seconds": 130.0,
        "max_question_chars": 2000,
        "allowed_evaluation_methods": ["b0", "b1", "m1", "m2", "b2", "m4"],
        "allowed_evaluation_splits": ["development", "validation"],
        "test_execution_status": "locked_until_s12",
    }


def _write(tmp_path: Path, **changes: object) -> Path:
    payload = _payload()
    payload.update(changes)
    path = tmp_path / "api.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_api_config_accepts_frozen_contract(tmp_path: Path) -> None:
    config = load_api_config(_write(tmp_path))

    assert config.api_version == "s9-api-v1"
    assert config.api_prefix == "/api/v1"
    assert config.allowed_origins == ("http://localhost:5173",)
    assert config.query_timeout_seconds == 130.0
    assert config.max_question_chars == 2000
    assert config.allowed_evaluation_splits == ("development", "validation")
    assert config.test_execution_status == "locked_until_s12"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"allowed_origins": ["*"]}, "wildcard"),
        ({"allowed_origins": ["http://localhost:5173/"]}, "origin"),
        ({"query_timeout_seconds": 0}, "query_timeout_seconds"),
        ({"max_question_chars": True}, "max_question_chars"),
        ({"allowed_evaluation_splits": ["development", "test"]}, "Test"),
        ({"test_execution_status": "open"}, "locked_until_s12"),
    ],
)
def test_load_api_config_rejects_unsafe_values(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        load_api_config(_write(tmp_path, **changes))


def test_load_api_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = _write(tmp_path, unexpected=True)

    with pytest.raises(ValueError, match="fields"):
        load_api_config(path)
