"""Strict loader for the versioned S2 chunking contract."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_MODEL_FIELDS = frozenset({"name", "revision", "max_length"})
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "chunking_version",
        "max_chunk_tokens",
        "overlap_tokens",
        "min_chunk_tokens",
        "query_budget",
        "special_tokens_budget",
        "safety_margin",
        "html_normalization_version",
        "id_rule_version",
        "embedding",
        "reranker",
    }
)


def _strict_fields(data: dict[str, Any], expected: frozenset[str], label: str) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    revision: str
    max_length: int

    @classmethod
    def from_dict(cls, data: object, field_name: str) -> "ModelSpec":
        if not isinstance(data, dict):
            raise TypeError(f"{field_name} must be an object")
        _strict_fields(data, _MODEL_FIELDS, field_name)
        name = _non_empty_string(data["name"], f"{field_name}.name")
        revision = _non_empty_string(data["revision"], f"{field_name}.revision")
        if _GIT_COMMIT_RE.fullmatch(revision) is None:
            raise ValueError(f"{field_name}.revision must be a lowercase commit SHA")
        max_length = _positive_int(data["max_length"], f"{field_name}.max_length")
        return cls(name=name, revision=revision, max_length=max_length)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    schema_version: int
    chunking_version: str
    max_chunk_tokens: int
    overlap_tokens: int
    min_chunk_tokens: int
    query_budget: int
    special_tokens_budget: int
    safety_margin: int
    html_normalization_version: str
    id_rule_version: str
    embedding: ModelSpec
    reranker: ModelSpec

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.chunking_version != "s2-chunk-v1":
            raise ValueError("chunking_version must be s2-chunk-v1")
        if self.html_normalization_version != "postgresql-html-v1":
            raise ValueError("html_normalization_version must be postgresql-html-v1")
        if self.id_rule_version != "sha256-nul-v1":
            raise ValueError("id_rule_version must be sha256-nul-v1")
        for field_name in (
            "max_chunk_tokens",
            "overlap_tokens",
            "min_chunk_tokens",
            "query_budget",
            "special_tokens_budget",
            "safety_margin",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.overlap_tokens >= self.max_chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than max_chunk_tokens")
        if self.min_chunk_tokens > self.max_chunk_tokens:
            raise ValueError("min_chunk_tokens must not exceed max_chunk_tokens")
        required = (
            self.max_chunk_tokens
            + self.query_budget
            + self.special_tokens_budget
            + self.safety_margin
        )
        available = min(self.embedding.max_length, self.reranker.max_length)
        if required > available:
            raise ValueError(
                f"combined token budget {required} exceeds model limit {available}"
            )


def load_chunking_config(path: Path) -> ChunkingConfig:
    """Load a config without accepting implicit defaults or extension fields."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("chunking config must be a JSON object")
    _strict_fields(data, _CONFIG_FIELDS, "chunking config")
    return ChunkingConfig(
        schema_version=data["schema_version"],
        chunking_version=_non_empty_string(
            data["chunking_version"], "chunking_version"
        ),
        max_chunk_tokens=data["max_chunk_tokens"],
        overlap_tokens=data["overlap_tokens"],
        min_chunk_tokens=data["min_chunk_tokens"],
        query_budget=data["query_budget"],
        special_tokens_budget=data["special_tokens_budget"],
        safety_margin=data["safety_margin"],
        html_normalization_version=_non_empty_string(
            data["html_normalization_version"], "html_normalization_version"
        ),
        id_rule_version=_non_empty_string(data["id_rule_version"], "id_rule_version"),
        embedding=ModelSpec.from_dict(data["embedding"], "embedding"),
        reranker=ModelSpec.from_dict(data["reranker"], "reranker"),
    )
