from __future__ import annotations

import json
from pathlib import Path

import pytest

from docguard.engineering.config import (
    load_engineering_config,
    verify_compatibility,
    verify_repository_compatibility,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "engineering_version": "s11-engineering-v1",
        "api_version": "s9-api-v1",
        "chunking_version": "s2-chunk-v1",
        "dense_index_version": "s3-dense-v1",
        "hybrid_version": "s5-hybrid-v1",
        "reranker_version": "s6-reranker-v1",
        "generation_version": "s7-deepseek-flash-v1",
        "refusal_version": "s8-refusal-v1",
        "test_execution_status": "locked_until_s12",
        "score_tolerance": 1e-6,
        "regression": {
            "min_recall_at_5": 0.5,
            "min_mrr_at_10": 0.4,
            "max_failed_queries": 0,
        },
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_engineering_config_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    payload = _payload()
    _write(path, payload)
    config = load_engineering_config(path)
    assert config.engineering_version == "s11-engineering-v1"
    assert config.score_tolerance == pytest.approx(1e-6)
    assert config.regression.min_recall_at_5 == pytest.approx(0.5)

    payload["unknown"] = True
    _write(path, payload)
    with pytest.raises(ValueError, match="fields"):
        load_engineering_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_execution_status", "unlocked"),
        ("score_tolerance", 0),
        ("dense_index_version", "latest"),
    ],
)
def test_load_engineering_config_rejects_unsafe_values(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "config.json"
    payload = _payload()
    payload[field] = value
    _write(path, payload)
    with pytest.raises(ValueError):
        load_engineering_config(path)


def test_verify_compatibility_rejects_cross_artifact_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "engineering.json"
    _write(config_path, _payload())
    config = load_engineering_config(config_path)
    identities = {
        "api_version": "s9-api-v1",
        "chunking_version": "s2-chunk-v1",
        "dense_index_version": "s3-dense-v1",
        "hybrid_version": "s5-hybrid-v1",
        "reranker_version": "s6-reranker-v1",
        "generation_version": "s7-deepseek-flash-v1",
        "refusal_version": "s8-refusal-v1",
        "test_execution_status": "locked_until_s12",
    }
    verify_compatibility(config, identities)
    identities["chunking_version"] = "s2-chunk-v2"
    with pytest.raises(ValueError, match="chunking_version"):
        verify_compatibility(config, identities)


def test_repository_engineering_config_is_compatible() -> None:
    root = Path(__file__).parents[2]
    config = load_engineering_config(
        root / "configs/engineering/s11-engineering-v1.json"
    )
    verify_compatibility(
        config,
        {
            "api_version": "s9-api-v1",
            "chunking_version": "s2-chunk-v1",
            "dense_index_version": "s3-dense-v1",
            "hybrid_version": "s5-hybrid-v1",
            "reranker_version": "s6-reranker-v1",
            "generation_version": "s7-deepseek-flash-v1",
            "refusal_version": "s8-refusal-v1",
            "test_execution_status": "locked_until_s12",
        },
    )
    verify_repository_compatibility(root, config)
