import copy
import json
from pathlib import Path

import pytest

from docguard.generation.config import (
    load_generation_config,
    normalize_base_url,
)
from docguard.generation.prompt import output_schema_sha256, prompt_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "generation" / "s7-generation-v1.json"
DEEPSEEK_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "generation" / "s7-deepseek-flash-v1.json"
)


def _config_dict() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "generation.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_frozen_s7_config_and_prompt_hashes() -> None:
    config = load_generation_config(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.generation_version == "s7-generation-v1"
    assert config.prompt_version == "s7-prompt-v1"
    assert config.protocol == "openai-chat-completions"
    assert config.default_base_url == "https://api.openai.com/v1"
    assert config.model == "gpt-5.4-mini-2026-03-17"
    assert config.response_format == "json_schema"
    assert config.reasoning_effort == "none"
    assert config.temperature == 0.0
    assert config.max_output_tokens == 800
    assert config.timeout_seconds == 60.0
    assert config.max_retries == 0
    assert config.evidence_top_k == 10
    assert config.evidence_max_chars == 4_000
    assert config.evidence_total_chars == 30_000
    assert config.generation_enabled is True
    assert config.citation_validation_enabled is True
    assert config.input_cost_per_million_usd == 0.75
    assert config.cached_input_cost_per_million_usd == 0.075
    assert config.output_cost_per_million_usd == 4.50
    assert config.pricing_checked_on == "2026-10-16"
    assert config.prompt_sha256 == prompt_sha256(config.prompt_version)
    assert config.output_schema_sha256 == output_schema_sha256()
    assert config.dense_config_sha256 == (
        "3e61d9b09969f6dc68244c128ef242b158f328f4de1dd8f4dde521625d7776c8"
    )
    assert config.dense_index_manifest_sha256 == (
        "ee039a4ee6f8859b534c2a765346e2b53997615772be0937dc217cbb96320f2d"
    )
    assert config.dataset_version == "s4-golden-v1"
    assert config.dataset_manifest_sha256 == (
        "68442c57bbe368369060b1e03d7fa415493c49d6c6f053b5ef53ae7323b0e1e5"
    )
    assert config.chunking_version == "s2-chunk-v1"
    assert config.chunks_sha256 == (
        "6aad4667593d6519657cd1823b713983e09ed9f819e48246e0db9c18f0694964"
    )


def test_frozen_deepseek_responses_config() -> None:
    config = load_generation_config(DEEPSEEK_CONFIG_PATH)

    assert config.provider == "deepseek"
    assert config.generation_version == "s7-deepseek-flash-v1"
    assert config.protocol == "deepseek-responses"
    assert config.default_base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-flash"
    assert config.documented_model_version == "DeepSeek-V4.1-Flash-1017"
    assert config.model_version_checked_on == "2026-10-18"
    assert config.response_format == "json_schema"
    assert config.reasoning_effort == "none"
    assert config.temperature == 0.0
    assert config.max_output_tokens == 800
    assert config.timeout_seconds == 120.0
    assert config.max_retries == 0
    assert config.minimum_input_cost_per_million_usd == 0.15
    assert config.input_cost_per_million_usd == 0.30
    assert config.minimum_cached_input_cost_per_million_usd == 0.003
    assert config.cached_input_cost_per_million_usd == 0.006
    assert config.minimum_output_cost_per_million_usd == 0.60
    assert config.output_cost_per_million_usd == 1.20
    assert config.pricing_checked_on == "2026-10-18"
    assert config.prompt_sha256 == prompt_sha256(config.prompt_version)
    assert config.output_schema_sha256 == output_schema_sha256()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider", "openai", "provider"),
        ("protocol", "openai-chat-completions", "protocol"),
        ("default_base_url", "https://example.com", "DeepSeek API root"),
        ("model", "deepseek-v4-pro", "deepseek-flash"),
        ("documented_model_version", "rolling", "documented_model_version"),
        ("model_version_checked_on", "2026-10-17", "model_version_checked_on"),
        ("reasoning_effort", "low", "reasoning_effort"),
        ("timeout_seconds", 60, "timeout_seconds"),
        ("minimum_input_cost_per_million_usd", 0.16, "minimum_input_cost"),
        ("input_cost_per_million_usd", 0.29, "input_cost"),
        ("minimum_cached_input_cost_per_million_usd", 0.004, "minimum_cached"),
        ("cached_input_cost_per_million_usd", 0.005, "cached_input_cost"),
        ("minimum_output_cost_per_million_usd", 0.61, "minimum_output_cost"),
        ("output_cost_per_million_usd", 1.19, "output_cost"),
        ("pricing_checked_on", "2026-10-17", "pricing_checked_on"),
    ],
)
def test_deepseek_config_rejects_frozen_identity_or_price_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = json.loads(DEEPSEEK_CONFIG_PATH.read_text(encoding="utf-8"))
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        load_generation_config(_write_config(tmp_path, payload))


def test_deepseek_config_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads(DEEPSEEK_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        load_generation_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version must be 1"),
        ("generation_version", "rolling", "generation_version"),
        ("model", "gpt-5.4-mini", "dated model snapshot"),
        ("response_format", "json_object", "response_format"),
        ("reasoning_effort", "medium", "reasoning_effort"),
        ("temperature", 0.2, "temperature"),
        ("max_output_tokens", 0, "max_output_tokens"),
        ("timeout_seconds", 0, "timeout_seconds"),
        ("max_retries", 1, "max_retries"),
        ("evidence_top_k", 9, "evidence_top_k"),
        ("evidence_max_chars", 0, "evidence_max_chars"),
        ("evidence_total_chars", 3999, "evidence_total_chars"),
        ("generation_enabled", 1, "generation_enabled"),
        ("citation_validation_enabled", 1, "citation_validation_enabled"),
        ("input_cost_per_million_usd", -1, "input_cost"),
        ("cached_input_cost_per_million_usd", -1, "cached_input_cost"),
        ("output_cost_per_million_usd", -1, "output_cost"),
        ("chunking_version", "latest", "chunking_version"),
        ("chunks_sha256", "0" * 64, "chunks_sha256"),
    ],
)
def test_generation_config_rejects_invalid_frozen_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _config_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        load_generation_config(_write_config(tmp_path, payload))


def test_generation_config_rejects_unknown_and_missing_fields(tmp_path: Path) -> None:
    payload = _config_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        load_generation_config(_write_config(tmp_path, payload))

    payload = _config_dict()
    payload.pop("model")
    with pytest.raises(ValueError, match="missing fields"):
        load_generation_config(_write_config(tmp_path, payload))


def test_generation_config_rejects_prompt_and_schema_hash_drift(tmp_path: Path) -> None:
    for field in ("prompt_sha256", "output_schema_sha256"):
        payload = copy.deepcopy(_config_dict())
        payload[field] = "0" * 64
        with pytest.raises(ValueError, match=field):
            load_generation_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    "value",
    [
        "http://api.openai.com/v1",
        "https://user@example.com/v1",
        "https://api.openai.com/v1?token=secret",
        "https://api.openai.com/v1#fragment",
        "https://api.openai.com/v1/chat/completions",
    ],
)
def test_normalize_base_url_rejects_unsafe_or_endpoint_urls(value: str) -> None:
    with pytest.raises(ValueError, match="base URL"):
        normalize_base_url(value)


def test_normalize_base_url_removes_one_or_more_trailing_slashes() -> None:
    assert normalize_base_url("https://api.openai.com/v1///") == (
        "https://api.openai.com/v1"
    )
