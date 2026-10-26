"""Strict immutable configuration for S6 Cross-Encoder reranking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "reranking_version",
        "rerank_enabled",
        "model",
        "tokenizer",
        "inference",
        "m1",
        "reranking",
        "chunking_version",
        "chunks_sha256",
    }
)
_MODEL_FIELDS = frozenset({"name", "revision", "manifest_path", "manifest_sha256"})
_TOKENIZER_FIELDS = frozenset({"path", "sha256", "max_length", "truncation"})
_INFERENCE_FIELDS = frozenset({"batch_size", "execution_provider"})
_M1_FIELDS = frozenset({"rrf_k", "dense_depth", "bm25_depth", "s5_config_sha256"})
_RERANKING_FIELDS = frozenset(
    {"depth_candidates", "output_depth", "selection_metrics"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _strict_fields(data: dict[str, Any], expected: frozenset[str], label: str) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    schema_version: int
    reranking_version: str
    rerank_enabled: bool
    model_name: str
    model_revision: str
    model_manifest_path: str
    model_manifest_sha256: str
    tokenizer_path: str
    tokenizer_sha256: str
    max_length: int
    truncation: str
    batch_size: int
    execution_provider: str
    rrf_k: int
    dense_depth: int
    bm25_depth: int
    s5_config_sha256: str
    rerank_depth_candidates: tuple[int, ...]
    output_depth: int
    selection_metrics: tuple[str, ...]
    chunking_version: str
    chunks_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> RerankerConfig:
        data = _mapping(value, "reranker config")
        _strict_fields(data, _TOP_FIELDS, "reranker config")
        model = _mapping(data["model"], "model")
        tokenizer = _mapping(data["tokenizer"], "tokenizer")
        inference = _mapping(data["inference"], "inference")
        m1 = _mapping(data["m1"], "m1")
        reranking = _mapping(data["reranking"], "reranking")
        _strict_fields(model, _MODEL_FIELDS, "model")
        _strict_fields(tokenizer, _TOKENIZER_FIELDS, "tokenizer")
        _strict_fields(inference, _INFERENCE_FIELDS, "inference")
        _strict_fields(m1, _M1_FIELDS, "m1")
        _strict_fields(reranking, _RERANKING_FIELDS, "reranking")

        depth_value = reranking["depth_candidates"]
        if not isinstance(depth_value, list) or not all(
            type(item) is int and item > 0 for item in depth_value
        ):
            raise ValueError("reranking.depth_candidates must be positive integers")
        metric_value = reranking["selection_metrics"]
        if not isinstance(metric_value, list) or not all(
            isinstance(item, str) and item for item in metric_value
        ):
            raise ValueError("reranking.selection_metrics must be non-empty strings")

        config = cls(
            schema_version=data["schema_version"],
            reranking_version=_string(data["reranking_version"], "reranking_version"),
            rerank_enabled=data["rerank_enabled"],
            model_name=_string(model["name"], "model.name"),
            model_revision=_string(model["revision"], "model.revision"),
            model_manifest_path=_string(model["manifest_path"], "model.manifest_path"),
            model_manifest_sha256=_sha256(
                model["manifest_sha256"], "model.manifest_sha256"
            ),
            tokenizer_path=_string(tokenizer["path"], "tokenizer.path"),
            tokenizer_sha256=_sha256(tokenizer["sha256"], "tokenizer.sha256"),
            max_length=_positive_int(tokenizer["max_length"], "tokenizer.max_length"),
            truncation=_string(tokenizer["truncation"], "tokenizer.truncation"),
            batch_size=_positive_int(inference["batch_size"], "inference.batch_size"),
            execution_provider=_string(
                inference["execution_provider"], "inference.execution_provider"
            ),
            rrf_k=_positive_int(m1["rrf_k"], "m1.rrf_k"),
            dense_depth=_positive_int(m1["dense_depth"], "m1.dense_depth"),
            bm25_depth=_positive_int(m1["bm25_depth"], "m1.bm25_depth"),
            s5_config_sha256=_sha256(m1["s5_config_sha256"], "m1.s5_config_sha256"),
            rerank_depth_candidates=tuple(depth_value),
            output_depth=_positive_int(reranking["output_depth"], "reranking.output_depth"),
            selection_metrics=tuple(metric_value),
            chunking_version=_string(data["chunking_version"], "chunking_version"),
            chunks_sha256=_sha256(data["chunks_sha256"], "chunks_sha256"),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.reranking_version != "s6-reranker-v1":
            raise ValueError("reranking_version must be s6-reranker-v1")
        if type(self.rerank_enabled) is not bool or not self.rerank_enabled:
            raise ValueError("rerank_enabled must be true")
        if self.model_name != "BAAI/bge-reranker-base":
            raise ValueError("model.name does not match the frozen model")
        if _REVISION_RE.fullmatch(self.model_revision) is None or self.model_revision != (
            "2cfc18c9415c912f9d8155881c133215df768a70"
        ):
            raise ValueError("model.revision does not match the frozen revision")
        if self.model_manifest_path != "data/manifests/s6-reranker-model.json":
            raise ValueError("model.manifest_path does not match the frozen path")
        if self.model_manifest_sha256 != (
            "a32c1fc79c0da13b8f09b68bc28a79bb444727e5f9dcbcff80bab720ae764b62"
        ):
            raise ValueError("model.manifest_sha256 does not match the frozen manifest")
        if self.tokenizer_path != (
            "data/tokenizers/9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e/"
            "tokenizer.json"
        ):
            raise ValueError("tokenizer.path does not match the frozen asset")
        if self.tokenizer_sha256 != (
            "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e"
        ):
            raise ValueError("tokenizer.sha256 does not match the frozen asset")
        if self.max_length != 512:
            raise ValueError("tokenizer.max_length must be 512")
        if self.truncation != "only_second":
            raise ValueError("tokenizer.truncation must be only_second")
        if self.batch_size != 8:
            raise ValueError("inference.batch_size must be 8")
        if self.execution_provider != "CPUExecutionProvider":
            raise ValueError("inference.execution_provider must be CPUExecutionProvider")
        if (self.rrf_k, self.dense_depth, self.bm25_depth) != (20, 50, 20):
            raise ValueError("M1 parameters must be rrf_k=20, dense=50, bm25=20")
        if self.s5_config_sha256 != (
            "f5c78ee94c27454bb278b4a9bb59e7ee8429bf3f4ea98f5c1b55cde9a0b380a2"
        ):
            raise ValueError("m1.s5_config_sha256 does not match the frozen S5 config")
        if self.rerank_depth_candidates != (10, 20, 50):
            raise ValueError("reranking.depth_candidates must be [10, 20, 50]")
        if self.output_depth != 10:
            raise ValueError("reranking.output_depth must be 10")
        if self.selection_metrics != (
            "ndcg_at_10",
            "mrr_at_10",
            "recall_at_10",
            "recall_at_5",
        ):
            raise ValueError("reranking.selection_metrics do not match the frozen order")
        if self.chunking_version != "s2-chunk-v1":
            raise ValueError("chunking_version must be s2-chunk-v1")


def load_reranker_config(path: Path) -> RerankerConfig:
    with path.open("r", encoding="utf-8") as file:
        return RerankerConfig.from_dict(json.load(file))
