from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docguard.retrieval.reranker_config import load_reranker_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "s6-reranker-v1.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "s6-reranker-model.json"


def test_committed_s6_config_freezes_reranking_contract() -> None:
    config = load_reranker_config(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.reranking_version == "s6-reranker-v1"
    assert config.rerank_enabled is True
    assert config.model_name == "BAAI/bge-reranker-base"
    assert config.model_revision == "2cfc18c9415c912f9d8155881c133215df768a70"
    assert config.model_manifest_path == "data/manifests/s6-reranker-model.json"
    assert config.model_manifest_sha256 == hashlib.sha256(
        MANIFEST_PATH.read_bytes()
    ).hexdigest()
    assert config.tokenizer_path == (
        "data/tokenizers/9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e/"
        "tokenizer.json"
    )
    assert config.tokenizer_sha256 == (
        "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e"
    )
    assert config.max_length == 512
    assert config.truncation == "only_second"
    assert config.batch_size == 8
    assert config.execution_provider == "CPUExecutionProvider"
    assert (config.rrf_k, config.dense_depth, config.bm25_depth) == (20, 50, 20)
    assert config.s5_config_sha256 == (
        "f5c78ee94c27454bb278b4a9bb59e7ee8429bf3f4ea98f5c1b55cde9a0b380a2"
    )
    assert config.rerank_depth_candidates == (10, 20, 50)
    assert config.output_depth == 10
    assert config.selection_metrics == (
        "ndcg_at_10",
        "mrr_at_10",
        "recall_at_10",
        "recall_at_5",
    )
    assert config.chunking_version == "s2-chunk-v1"
    assert config.chunks_sha256 == (
        "6aad4667593d6519657cd1823b713983e09ed9f819e48246e0db9c18f0694964"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("extra", 1), "unknown fields"),
        (lambda value: value.pop("model"), "missing fields"),
        (lambda value: value.__setitem__("rerank_enabled", 1), "must be true"),
        (lambda value: value["model"].__setitem__("revision", "main"), "revision"),
        (
            lambda value: value["tokenizer"].__setitem__("max_length", 256),
            "max_length",
        ),
        (
            lambda value: value["tokenizer"].__setitem__("truncation", "longest_first"),
            "only_second",
        ),
        (
            lambda value: value["inference"].__setitem__("batch_size", True),
            "batch_size",
        ),
        (
            lambda value: value["m1"].__setitem__("rrf_k", 60),
            "M1 parameters",
        ),
        (
            lambda value: value["reranking"].__setitem__(
                "depth_candidates", [20, 10, 50]
            ),
            "depth_candidates",
        ),
        (
            lambda value: value["reranking"].__setitem__("output_depth", 20),
            "output_depth",
        ),
        (lambda value: value.__setitem__("chunks_sha256", "bad"), "SHA-256"),
    ],
)
def test_s6_config_rejects_contract_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutation(value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_reranker_config(path)


def test_s6_config_rejects_manifest_hash_drift(tmp_path: Path) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["model"]["manifest_sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_sha256"):
        load_reranker_config(path)
