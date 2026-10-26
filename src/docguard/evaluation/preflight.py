"""Fail-closed S12 preflight inspection without executing Test methods."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from docguard.evaluation.dataset import sha256_file
from docguard.evaluation.final_protocol import (
    S12FinalProtocol,
    load_final_protocol,
    verify_final_protocol_inputs,
)
from docguard.evaluation.runner import _atomic_text, _json_line
from docguard.retrieval.bm25_index import bm25_index_stats
from docguard.retrieval.config import load_dense_config
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.reranker_config import load_reranker_config

_ASSET_NAMES = {
    "chunks",
    "dense_index",
    "bm25_index",
    "embedding_model",
    "reranker_model",
    "tokenizers",
}


class S12PreflightError(RuntimeError):
    """Raised before a seal is written when S12 is unsafe to execute."""


def validate_preflight_state(state: Mapping[str, object]) -> None:
    """Validate all execution prerequisites without accepting partial success."""
    if state.get("git_clean") is not True:
        raise S12PreflightError("working tree must be clean before sealing S12")
    if state.get("credential_present") is not True:
        raise S12PreflightError("DeepSeek credential is unavailable")
    if state.get("old_seal_present") is not False:
        raise S12PreflightError("an old S12 preflight seal already exists")
    if state.get("existing_output_dirs") != []:
        raise S12PreflightError(
            "one or more formal S12 output directories already exist"
        )
    dataset = state.get("dataset")
    if not isinstance(dataset, Mapping) or dataset != {
        "status": "frozen",
        "human_review_complete": True,
        "test_double_review_complete": True,
        "test_execution_status": "locked_until_s12",
        "test_review_count": 60,
    }:
        raise S12PreflightError("dataset review contract is incomplete")
    assets = state.get("assets")
    if (
        not isinstance(assets, Mapping)
        or set(assets) != _ASSET_NAMES
        or any(value is not True for value in assets.values())
    ):
        raise S12PreflightError("asset verification is incomplete")


def _git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _credential_present(environment: Mapping[str, str]) -> bool:
    if environment.get("DEEPSEEK_API_KEY", "").strip():
        return True
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "DEEPSEEK_API_KEY")
        return isinstance(value, str) and bool(value.strip())
    except (FileNotFoundError, OSError):
        return False


async def _dense_index_details(
    asset_root: Path, config_path: Path, chunks_sha256: str, model_manifest_sha256: str
) -> dict[str, object]:
    config = load_dense_config(config_path)
    index_root = asset_root / ".qdrant/s3-dense-v1"
    build_info = json.loads(
        (index_root / "build-info.json").read_text(encoding="utf-8")
    )
    expected = {
        "chunks_sha256": chunks_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "point_count": 7413,
        "version_counts": {"17.11": 3670, "18.6": 3743},
        "collection_name": config.collection_name,
        "vector_size": config.model.embedding_dimension,
        "distance": config.distance,
    }
    if not isinstance(build_info, dict) or any(
        build_info.get(name) != value for name, value in expected.items()
    ):
        raise S12PreflightError("dense index build information is incompatible")
    client = AsyncQdrantClient(path=str(index_root / "qdrant"))
    try:
        if not await client.collection_exists(config.collection_name):
            raise S12PreflightError("dense index collection is unavailable")
        count = await client.count(config.collection_name, exact=True)
        info = await client.get_collection(config.collection_name)
        vectors = info.config.params.vectors
        if count.count != 7413 or not isinstance(vectors, dict):
            raise S12PreflightError("dense index count or schema is incompatible")
        dense = vectors.get("dense")
        if (
            dense is None
            or dense.size != config.model.embedding_dimension
            or dense.distance.value != config.distance
        ):
            raise S12PreflightError("dense vector schema is incompatible")
    finally:
        await client.close()
    return {
        "build_info_sha256": sha256_file(index_root / "build-info.json"),
        "point_count": 7413,
        "version_counts": expected["version_counts"],
    }


async def _verify_assets(
    root: Path, asset_root: Path
) -> tuple[dict[str, bool], dict[str, object]]:
    chunks = asset_root / "data/processed/s2-chunk-v1/chunks.jsonl"
    dense_config_path = root / "configs/retrieval/s3-dense-v1.json"
    reranker_config_path = root / "configs/retrieval/s6-reranker-v1.json"
    dense_config = load_dense_config(dense_config_path)
    reranker_config = load_reranker_config(reranker_config_path)
    if sha256_file(chunks) != dense_config.chunks_sha256:
        raise S12PreflightError("chunks hash differs from the frozen dense config")

    dense_manifest = root / "data/manifests/s3-embedding-model.json"
    reranker_manifest = root / reranker_config.model_manifest_path
    dense_assets = verify_model_assets(
        asset_root / "data/models/s3-e5-small",
        dense_manifest,
        expected_model_name=dense_config.model.name,
        expected_revision=dense_config.model.revision,
    )
    reranker_assets = verify_model_assets(
        asset_root / "data/models/s6-bge-reranker-base",
        reranker_manifest,
        expected_model_name=reranker_config.model_name,
        expected_revision=reranker_config.model_revision,
    )
    dense_tokenizer = root / (
        "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/"
        "tokenizer.json"
    )
    reranker_tokenizer = root / reranker_config.tokenizer_path
    if sha256_file(dense_tokenizer) != dense_tokenizer.parent.name:
        raise S12PreflightError("dense tokenizer hash is incompatible")
    if sha256_file(reranker_tokenizer) != reranker_config.tokenizer_sha256:
        raise S12PreflightError("reranker tokenizer hash is incompatible")

    dense_details = await _dense_index_details(
        asset_root,
        dense_config_path,
        dense_config.chunks_sha256,
        sha256_file(dense_manifest),
    )
    bm25 = bm25_index_stats(asset_root / ".bm25/s5-bm25-v1")
    if bm25.document_count != 7413 or bm25.version_counts != {
        "17.11": 3670,
        "18.6": 3743,
    }:
        raise S12PreflightError("BM25 index counts are incompatible")
    flags = {name: True for name in sorted(_ASSET_NAMES)}
    details: dict[str, object] = {
        "chunks_sha256": dense_config.chunks_sha256,
        "embedding_model": {
            "file_count": dense_assets.file_count,
            "total_size_bytes": dense_assets.total_size_bytes,
            "manifest_sha256": sha256_file(dense_manifest),
        },
        "reranker_model": {
            "file_count": reranker_assets.file_count,
            "total_size_bytes": reranker_assets.total_size_bytes,
            "manifest_sha256": sha256_file(reranker_manifest),
        },
        "dense_index": dense_details,
        "bm25_index": {
            "document_count": bm25.document_count,
            "logical_sha256": bm25.logical_sha256,
            "version_counts": bm25.version_counts,
        },
        "tokenizer_sha256": {
            "dense": sha256_file(dense_tokenizer),
            "reranker": sha256_file(reranker_tokenizer),
        },
    }
    return flags, details


def _dataset_review_state(root: Path, protocol: S12FinalProtocol) -> dict[str, object]:
    manifest = json.loads(
        (root / protocol.dataset.manifest_path).read_text(encoding="utf-8")
    )
    review_inputs = (
        manifest.get("review_inputs") if isinstance(manifest, dict) else None
    )
    second = (
        review_inputs.get("reviews_test_round2.jsonl")
        if isinstance(review_inputs, dict)
        else None
    )
    return {
        "status": manifest.get("status") if isinstance(manifest, dict) else None,
        "human_review_complete": (
            manifest.get("human_review_complete")
            if isinstance(manifest, dict)
            else None
        ),
        "test_double_review_complete": (
            manifest.get("test_double_review_complete")
            if isinstance(manifest, dict)
            else None
        ),
        "test_execution_status": (
            manifest.get("test_execution_status")
            if isinstance(manifest, dict)
            else None
        ),
        "test_review_count": second.get("count") if isinstance(second, dict) else None,
    }


async def prepare_s12_preflight(
    root: Path,
    asset_root: Path,
    output_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Inspect and seal an exact clean commit; never run retrieval or generation."""
    root = root.resolve()
    asset_root = asset_root.resolve()
    output_path = output_path.resolve()
    expected_output = (root / "reports/s12_final/preflight.json").resolve()
    if output_path != expected_output:
        raise S12PreflightError("preflight seal path must use the frozen S12 location")
    protocol_path = root / "configs/evaluation/s12-final-v1.json"
    protocol = load_final_protocol(protocol_path)
    verify_final_protocol_inputs(root, protocol)
    status = _git_value(root, "status", "--porcelain=v1")
    existing = [
        name
        for name in protocol.method_order
        if os.path.lexists(
            root / protocol.output_root / protocol.methods[name].output_dir
        )
    ]
    asset_flags, asset_details = await _verify_assets(root, asset_root)
    state: dict[str, object] = {
        "git_clean": status == "",
        "credential_present": _credential_present(environment or os.environ),
        "old_seal_present": os.path.lexists(output_path),
        "existing_output_dirs": existing,
        "dataset": _dataset_review_state(root, protocol),
        "assets": asset_flags,
    }
    validate_preflight_state(state)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": protocol.protocol_version,
        "status": "sealed",
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_sha256": protocol.dataset.test_sha256,
        "question_count": protocol.dataset.test_count,
        "review_status": protocol.dataset.review_status,
        "method_order": list(protocol.method_order),
        "max_retries": protocol.max_retries,
        "existing_record_policy": protocol.existing_record_policy,
        "results_viewed": False,
        "credential_present": True,
        "assets": asset_details,
    }
    _atomic_text(output_path, _json_line(payload))
    return payload


def run_preflight(root: Path, asset_root: Path, output_path: Path) -> dict[str, object]:
    return asyncio.run(prepare_s12_preflight(root, asset_root, output_path))
