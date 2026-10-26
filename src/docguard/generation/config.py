"""Strict loaders for the frozen S7 generation configurations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from docguard.fs_safety import validate_regular_file
from docguard.generation.prompt import output_schema_sha256, prompt_sha256

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MODEL_SNAPSHOT_RE = re.compile(r"gpt-[a-z0-9.-]+-\d{4}-\d{2}-\d{2}")
_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "generation_version",
        "prompt_version",
        "prompt_sha256",
        "output_schema_sha256",
        "protocol",
        "default_base_url",
        "model",
        "response_format",
        "reasoning_effort",
        "temperature",
        "max_output_tokens",
        "timeout_seconds",
        "max_retries",
        "evidence_top_k",
        "evidence_max_chars",
        "evidence_total_chars",
        "generation_enabled",
        "citation_validation_enabled",
        "input_cost_per_million_usd",
        "cached_input_cost_per_million_usd",
        "output_cost_per_million_usd",
        "pricing_checked_on",
        "dense_config_sha256",
        "dense_index_manifest_sha256",
        "dataset_version",
        "dataset_manifest_sha256",
        "chunking_version",
        "corpus_manifest_sha256",
        "chunks_sha256",
    }
)
_DEEPSEEK_ONLY_FIELDS = frozenset(
    {
        "provider",
        "documented_model_version",
        "model_version_checked_on",
        "minimum_input_cost_per_million_usd",
        "minimum_cached_input_cost_per_million_usd",
        "minimum_output_cost_per_million_usd",
    }
)
_EXPECTED_HASHES = {
    "dense_config_sha256": "3e61d9b09969f6dc68244c128ef242b158f328f4de1dd8f4dde521625d7776c8",
    "dense_index_manifest_sha256": "ee039a4ee6f8859b534c2a765346e2b53997615772be0937dc217cbb96320f2d",
    "dataset_manifest_sha256": "68442c57bbe368369060b1e03d7fa415493c49d6c6f053b5ef53ae7323b0e1e5",
    "corpus_manifest_sha256": "0559ee4d75d6079fc2859f8e09910570c4f37f631511f0a9f4348ae7023911d2",
    "chunks_sha256": "6aad4667593d6519657cd1823b713983e09ed9f819e48246e0db9c18f0694964",
}


def normalize_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base URL must be a non-empty HTTPS URL")
    parsed = urlsplit(value.strip())
    path = parsed.path.rstrip("/")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or path.endswith(("/chat/completions", "/responses"))
    ):
        raise ValueError("base URL must be a safe HTTPS API root")
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit(("https", netloc, path, "", ""))


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    schema_version: int
    provider: str
    generation_version: str
    prompt_version: str
    prompt_sha256: str
    output_schema_sha256: str
    protocol: str
    default_base_url: str
    model: str
    documented_model_version: str | None
    model_version_checked_on: str | None
    response_format: str
    reasoning_effort: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    max_retries: int
    evidence_top_k: int
    evidence_max_chars: int
    evidence_total_chars: int
    generation_enabled: bool
    citation_validation_enabled: bool
    minimum_input_cost_per_million_usd: float
    input_cost_per_million_usd: float
    minimum_cached_input_cost_per_million_usd: float
    cached_input_cost_per_million_usd: float
    minimum_output_cost_per_million_usd: float
    output_cost_per_million_usd: float
    pricing_checked_on: str
    dense_config_sha256: str
    dense_index_manifest_sha256: str
    dataset_version: str
    dataset_manifest_sha256: str
    chunking_version: str
    corpus_manifest_sha256: str
    chunks_sha256: str


def _strict_fields(payload: dict[str, Any], expected: frozenset[str]) -> None:
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise ValueError(
            f"generation config contains unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise ValueError(f"generation config is missing fields: {sorted(missing)}")


def _require_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_number(value: object, field: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if (positive and numeric <= 0) or (not positive and numeric < 0):
        raise ValueError(f"{field} has an invalid value")
    return numeric


def _common_values(payload: dict[str, Any]) -> dict[str, object]:
    if payload["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if payload["prompt_version"] != "s7-prompt-v1":
        raise ValueError("prompt_version must be s7-prompt-v1")
    saved_prompt_hash = _sha256(payload["prompt_sha256"], "prompt_sha256")
    if saved_prompt_hash != prompt_sha256(payload["prompt_version"]):
        raise ValueError("prompt_sha256 does not match the frozen prompt")
    saved_schema_hash = _sha256(payload["output_schema_sha256"], "output_schema_sha256")
    if saved_schema_hash != output_schema_sha256():
        raise ValueError("output_schema_sha256 does not match the frozen schema")
    if payload["response_format"] != "json_schema":
        raise ValueError("response_format must be json_schema")
    if payload["reasoning_effort"] != "none":
        raise ValueError("reasoning_effort must be none")
    temperature = _require_number(payload["temperature"], "temperature")
    if temperature != 0:
        raise ValueError("temperature must be 0")
    max_output_tokens = _require_positive_int(
        payload["max_output_tokens"], "max_output_tokens"
    )
    if max_output_tokens != 800:
        raise ValueError("max_output_tokens must be 800")
    if payload["max_retries"] != 0:
        raise ValueError("max_retries must be 0")
    evidence_top_k = _require_positive_int(payload["evidence_top_k"], "evidence_top_k")
    if evidence_top_k != 10:
        raise ValueError("evidence_top_k must be 10")
    evidence_max_chars = _require_positive_int(
        payload["evidence_max_chars"], "evidence_max_chars"
    )
    evidence_total_chars = _require_positive_int(
        payload["evidence_total_chars"], "evidence_total_chars"
    )
    if evidence_total_chars < evidence_max_chars:
        raise ValueError("evidence_total_chars must cover at least one evidence item")
    hashes = {field: _sha256(payload[field], field) for field in _EXPECTED_HASHES}
    for field, expected in _EXPECTED_HASHES.items():
        if hashes[field] != expected:
            raise ValueError(f"{field} does not match the frozen upstream identity")
    if payload["dataset_version"] != "s4-golden-v1":
        raise ValueError("dataset_version must be s4-golden-v1")
    if payload["chunking_version"] != "s2-chunk-v1":
        raise ValueError("chunking_version must be s2-chunk-v1")
    return {
        "prompt_sha256": saved_prompt_hash,
        "output_schema_sha256": saved_schema_hash,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "evidence_top_k": evidence_top_k,
        "evidence_max_chars": evidence_max_chars,
        "evidence_total_chars": evidence_total_chars,
        "generation_enabled": _require_bool(
            payload["generation_enabled"], "generation_enabled"
        ),
        "citation_validation_enabled": _require_bool(
            payload["citation_validation_enabled"], "citation_validation_enabled"
        ),
        **hashes,
    }


def _require_frozen_price(
    payload: dict[str, Any], field: str, expected: float
) -> float:
    value = _require_number(payload[field], field)
    if value != expected:
        raise ValueError(f"{field} must be {expected}")
    return value


def _build_config(
    payload: dict[str, Any],
    common: dict[str, object],
    *,
    provider: str,
    base_url: str,
    model: str,
    documented_model_version: str | None,
    model_version_checked_on: str | None,
    timeout: float,
    minimum_prices: dict[str, float],
    maximum_prices: dict[str, float],
) -> GenerationConfig:
    return GenerationConfig(
        schema_version=1,
        provider=provider,
        generation_version=payload["generation_version"],
        prompt_version=payload["prompt_version"],
        prompt_sha256=common["prompt_sha256"],
        output_schema_sha256=common["output_schema_sha256"],
        protocol=payload["protocol"],
        default_base_url=base_url,
        model=model,
        documented_model_version=documented_model_version,
        model_version_checked_on=model_version_checked_on,
        response_format=payload["response_format"],
        reasoning_effort=payload["reasoning_effort"],
        temperature=common["temperature"],
        max_output_tokens=common["max_output_tokens"],
        timeout_seconds=timeout,
        max_retries=0,
        evidence_top_k=common["evidence_top_k"],
        evidence_max_chars=common["evidence_max_chars"],
        evidence_total_chars=common["evidence_total_chars"],
        generation_enabled=common["generation_enabled"],
        citation_validation_enabled=common["citation_validation_enabled"],
        minimum_input_cost_per_million_usd=minimum_prices["input"],
        input_cost_per_million_usd=maximum_prices["input"],
        minimum_cached_input_cost_per_million_usd=minimum_prices["cached"],
        cached_input_cost_per_million_usd=maximum_prices["cached"],
        minimum_output_cost_per_million_usd=minimum_prices["output"],
        output_cost_per_million_usd=maximum_prices["output"],
        pricing_checked_on=payload["pricing_checked_on"],
        dense_config_sha256=common["dense_config_sha256"],
        dense_index_manifest_sha256=common["dense_index_manifest_sha256"],
        dataset_version=payload["dataset_version"],
        dataset_manifest_sha256=common["dataset_manifest_sha256"],
        chunking_version=payload["chunking_version"],
        corpus_manifest_sha256=common["corpus_manifest_sha256"],
        chunks_sha256=common["chunks_sha256"],
    )


def _load_openai(payload: dict[str, Any]) -> GenerationConfig:
    _strict_fields(payload, _COMMON_FIELDS)
    common = _common_values(payload)
    if payload["generation_version"] != "s7-generation-v1":
        raise ValueError("generation_version must be s7-generation-v1")
    if payload["protocol"] != "openai-chat-completions":
        raise ValueError("protocol must be openai-chat-completions")
    base_url = normalize_base_url(payload["default_base_url"])
    if base_url != "https://api.openai.com/v1":
        raise ValueError("default_base_url must match the frozen OpenAI API root")
    model = payload["model"]
    if not isinstance(model, str) or _MODEL_SNAPSHOT_RE.fullmatch(model) is None:
        raise ValueError("model must be a dated model snapshot")
    if model != "gpt-5.4-mini-2026-03-17":
        raise ValueError("model must match the frozen S7 snapshot")
    timeout = _require_number(
        payload["timeout_seconds"], "timeout_seconds", positive=True
    )
    if timeout != 60:
        raise ValueError("timeout_seconds must be 60")
    prices = {
        "input": _require_frozen_price(payload, "input_cost_per_million_usd", 0.75),
        "cached": _require_frozen_price(
            payload, "cached_input_cost_per_million_usd", 0.075
        ),
        "output": _require_frozen_price(payload, "output_cost_per_million_usd", 4.5),
    }
    if payload["pricing_checked_on"] != "2026-10-16":
        raise ValueError("pricing_checked_on must be 2026-10-16")
    return _build_config(
        payload,
        common,
        provider="openai",
        base_url=base_url,
        model=model,
        documented_model_version=None,
        model_version_checked_on=None,
        timeout=timeout,
        minimum_prices=prices,
        maximum_prices=prices,
    )


def _load_deepseek(payload: dict[str, Any]) -> GenerationConfig:
    _strict_fields(payload, _COMMON_FIELDS | _DEEPSEEK_ONLY_FIELDS)
    common = _common_values(payload)
    if payload["provider"] != "deepseek":
        raise ValueError("provider must be deepseek")
    if payload["generation_version"] != "s7-deepseek-flash-v1":
        raise ValueError("generation_version must be s7-deepseek-flash-v1")
    if payload["protocol"] != "deepseek-responses":
        raise ValueError("protocol must be deepseek-responses")
    base_url = normalize_base_url(payload["default_base_url"])
    if base_url != "https://api.deepseek.com":
        raise ValueError("default_base_url must match the frozen DeepSeek API root")
    if payload["model"] != "deepseek-flash":
        raise ValueError("model must be deepseek-flash")
    if payload["documented_model_version"] != "DeepSeek-V4.1-Flash-1017":
        raise ValueError(
            "documented_model_version must match the checked official docs"
        )
    if payload["model_version_checked_on"] != "2026-10-18":
        raise ValueError("model_version_checked_on must be 2026-10-18")
    timeout = _require_number(
        payload["timeout_seconds"], "timeout_seconds", positive=True
    )
    if timeout != 120:
        raise ValueError("timeout_seconds must be 120")
    minimum_prices = {
        "input": _require_frozen_price(
            payload, "minimum_input_cost_per_million_usd", 0.15
        ),
        "cached": _require_frozen_price(
            payload, "minimum_cached_input_cost_per_million_usd", 0.003
        ),
        "output": _require_frozen_price(
            payload, "minimum_output_cost_per_million_usd", 0.6
        ),
    }
    maximum_prices = {
        "input": _require_frozen_price(payload, "input_cost_per_million_usd", 0.3),
        "cached": _require_frozen_price(
            payload, "cached_input_cost_per_million_usd", 0.006
        ),
        "output": _require_frozen_price(payload, "output_cost_per_million_usd", 1.2),
    }
    if payload["pricing_checked_on"] != "2026-10-18":
        raise ValueError("pricing_checked_on must be 2026-10-18")
    return _build_config(
        payload,
        common,
        provider="deepseek",
        base_url=base_url,
        model="deepseek-flash",
        documented_model_version="DeepSeek-V4.1-Flash-1017",
        model_version_checked_on="2026-10-18",
        timeout=timeout,
        minimum_prices=minimum_prices,
        maximum_prices=maximum_prices,
    )


def load_generation_config(path: Path) -> GenerationConfig:
    validate_regular_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("generation config must be a JSON object")
    if "provider" in payload:
        return _load_deepseek(payload)
    return _load_openai(payload)
