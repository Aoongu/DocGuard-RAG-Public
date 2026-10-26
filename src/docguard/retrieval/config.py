"""Strict loader for the frozen S3 dense retrieval contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "index_version",
        "dense_enabled",
        "model",
        "batch_size",
        "collection_name",
        "distance",
        "default_top_k",
        "chunking_version",
        "s2_manifest_sha256",
        "chunks_sha256",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "name",
        "revision",
        "embedding_dimension",
        "max_length",
        "query_prefix",
        "passage_prefix",
        "normalize_embeddings",
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


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class DenseModelSpec:
    name: str
    revision: str
    embedding_dimension: int
    max_length: int
    query_prefix: str
    passage_prefix: str
    normalize_embeddings: bool

    @classmethod
    def from_dict(cls, value: object) -> DenseModelSpec:
        if not isinstance(value, dict):
            raise TypeError("model must be an object")
        _strict_fields(value, _MODEL_FIELDS, "model")
        model = cls(
            name=_string(value["name"], "model.name"),
            revision=_string(value["revision"], "model.revision"),
            embedding_dimension=_positive_int(
                value["embedding_dimension"], "model.embedding_dimension"
            ),
            max_length=_positive_int(value["max_length"], "model.max_length"),
            query_prefix=_string(value["query_prefix"], "model.query_prefix"),
            passage_prefix=_string(value["passage_prefix"], "model.passage_prefix"),
            normalize_embeddings=value["normalize_embeddings"],
        )
        if model.name != "intfloat/multilingual-e5-small":
            raise ValueError("model.name must be intfloat/multilingual-e5-small")
        if _REVISION_RE.fullmatch(model.revision) is None:
            raise ValueError("model.revision must be a lowercase commit SHA")
        if model.embedding_dimension != 384:
            raise ValueError("model.embedding_dimension must be 384")
        if model.max_length != 512:
            raise ValueError("model.max_length must be 512")
        if model.query_prefix != "query: ":
            raise ValueError("model.query_prefix must be 'query: '")
        if model.passage_prefix != "passage: ":
            raise ValueError("model.passage_prefix must be 'passage: '")
        if type(model.normalize_embeddings) is not bool or not model.normalize_embeddings:
            raise ValueError("model.normalize_embeddings must be true")
        return model


@dataclass(frozen=True, slots=True)
class DenseConfig:
    schema_version: int
    index_version: str
    dense_enabled: bool
    model: DenseModelSpec
    batch_size: int
    collection_name: str
    distance: str
    default_top_k: int
    chunking_version: str
    s2_manifest_sha256: str
    chunks_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.index_version != "s3-dense-v1":
            raise ValueError("index_version must be s3-dense-v1")
        if type(self.dense_enabled) is not bool:
            raise ValueError("dense_enabled must be a boolean")
        if not self.dense_enabled:
            raise ValueError("the frozen baseline requires dense_enabled=true")
        if self.batch_size != 32:
            raise ValueError("batch_size must be 32")
        if self.collection_name != "docguard_postgresql_s2_chunk_v1_e5_small":
            raise ValueError("collection_name does not match the frozen S3 index")
        if self.distance != "Cosine":
            raise ValueError("distance must be Cosine")
        if self.default_top_k != 5:
            raise ValueError("default_top_k must be 5")
        if self.chunking_version != "s2-chunk-v1":
            raise ValueError("chunking_version must be s2-chunk-v1")
        _sha256(self.s2_manifest_sha256, "s2_manifest_sha256")
        _sha256(self.chunks_sha256, "chunks_sha256")


def load_dense_config(path: Path) -> DenseConfig:
    """Load the exact S3 configuration without implicit defaults."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("dense config must be a JSON object")
    _strict_fields(data, _CONFIG_FIELDS, "dense config")
    return DenseConfig(
        schema_version=data["schema_version"],
        index_version=_string(data["index_version"], "index_version"),
        dense_enabled=data["dense_enabled"],
        model=DenseModelSpec.from_dict(data["model"]),
        batch_size=data["batch_size"],
        collection_name=_string(data["collection_name"], "collection_name"),
        distance=_string(data["distance"], "distance"),
        default_top_k=data["default_top_k"],
        chunking_version=_string(data["chunking_version"], "chunking_version"),
        s2_manifest_sha256=_sha256(
            data["s2_manifest_sha256"], "s2_manifest_sha256"
        ),
        chunks_sha256=_sha256(data["chunks_sha256"], "chunks_sha256"),
    )
