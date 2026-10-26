from __future__ import annotations

import json
import math
import os
import socket
from pathlib import Path

import pytest

from docguard.ingestion.catalog import SOURCES
from docguard.ingestion.snapshot import load_snapshot_manifest, restore_snapshot
from docguard.processing.pipeline import build_corpus
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.pipeline import build_index, verify_index
from docguard.retrieval.service import create_default_retriever, run_smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_MANIFEST = PROJECT_ROOT / "data/manifests/snapshot.jsonl"
S2_CONFIG = PROJECT_ROOT / "configs/chunking/s2-chunk-v1.json"
S2_TOKENIZERS = PROJECT_ROOT / "data/tokenizers/manifest.json"
S3_CONFIG = PROJECT_ROOT / "configs/retrieval/s3-dense-v1.json"
MODEL_ROOT = PROJECT_ROOT / "data/models/s3-e5-small"
MODEL_MANIFEST = PROJECT_ROOT / "data/manifests/s3-embedding-model.json"
INDEX_RECEIPT = PROJECT_ROOT / "data/manifests/s3-dense-v1.json"
TOKENIZER = PROJECT_ROOT / (
    "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/"
    "tokenizer.json"
)
SMOKE_QUERIES = PROJECT_ROOT / "data/eval/s3-smoke-v1.jsonl"
SMOKE_RESULTS = PROJECT_ROOT / "reports/s3_dense_smoke_results.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_same_results(
    expected: list[dict[str, object]], actual: list[dict[str, object]]
) -> None:
    assert len(expected) == len(actual) == 10
    for expected_row, actual_row in zip(expected, actual, strict=True):
        for key in ("schema_version", "query_id", "query", "version", "top_k"):
            assert expected_row[key] == actual_row[key]
        expected_hits = expected_row["hits"]
        actual_hits = actual_row["hits"]
        assert isinstance(expected_hits, list) and isinstance(actual_hits, list)
        assert len(expected_hits) == len(actual_hits) == 5
        for expected_hit, actual_hit in zip(expected_hits, actual_hits, strict=True):
            assert isinstance(expected_hit, dict) and isinstance(actual_hit, dict)
            assert expected_hit.keys() == actual_hit.keys()
            for key in expected_hit.keys() - {"score"}:
                assert expected_hit[key] == actual_hit[key]
            assert math.isfinite(expected_hit["score"])
            assert actual_hit["score"] == pytest.approx(
                expected_hit["score"], rel=1e-6, abs=1e-7
            )
            if expected_row["version"] is not None:
                assert expected_hit["postgresql_version"] == expected_row["version"]


@pytest.mark.skipif(
    os.environ.get("DOCGUARD_RUN_S3_INTEGRATION") != "1",
    reason="set DOCGUARD_RUN_S3_INTEGRATION=1 for two full offline index rebuilds",
)
@pytest.mark.asyncio
async def test_official_dense_index_is_offline_traceable_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert MODEL_MANIFEST.is_file()
    assert INDEX_RECEIPT.is_file()
    assert SMOKE_RESULTS.is_file()
    assert not list(PROJECT_ROOT.glob("reports/*s3*.csv"))
    verify_model_assets(MODEL_ROOT, MODEL_MANIFEST)
    committed_receipt = json.loads(INDEX_RECEIPT.read_text(encoding="utf-8"))
    committed_results = _read_jsonl(SMOKE_RESULTS)

    def forbid_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("S3 official build must remain offline")

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    raw_root = tmp_path / "raw"
    restore_snapshot(SOURCES, load_snapshot_manifest(SNAPSHOT_MANIFEST), raw_root)
    corpus_root = tmp_path / "s2-chunk-v1"
    corpus = build_corpus(
        raw_root,
        corpus_root,
        config_path=S2_CONFIG,
        tokenizer_manifest_path=S2_TOKENIZERS,
        snapshot_manifest_path=SNAPSHOT_MANIFEST,
    )
    assert corpus.chunk_count == 7413

    rebuilt_outputs = []
    for name in ("first", "second"):
        index_root = tmp_path / name / "s3-dense-v1"
        result = await build_index(
            corpus_root / "chunks.jsonl",
            index_root,
            config_path=S3_CONFIG,
            model_root=MODEL_ROOT,
            model_manifest_path=MODEL_MANIFEST,
            tokenizer_path=TOKENIZER,
        )
        verified = await verify_index(
            corpus_root / "chunks.jsonl",
            index_root,
            config_path=S3_CONFIG,
            model_root=MODEL_ROOT,
            model_manifest_path=MODEL_MANIFEST,
            tokenizer_path=TOKENIZER,
        )
        assert result.status == "built"
        assert result.point_count == verified.point_count == 7413
        assert sum(result.version_counts.values()) == 7413
        assert json.loads((index_root / "build-info.json").read_text(encoding="utf-8")) == (
            committed_receipt
        )

        retriever = create_default_retriever(
            index_root=index_root,
            config_path=S3_CONFIG,
            model_root=MODEL_ROOT,
            model_manifest_path=MODEL_MANIFEST,
            tokenizer_path=TOKENIZER,
        )
        output = tmp_path / f"{name}-smoke.jsonl"
        try:
            assert await run_smoke(SMOKE_QUERIES, output, retriever) == 10
        finally:
            await retriever.close()
        rebuilt_outputs.append(_read_jsonl(output))

    _assert_same_results(committed_results, rebuilt_outputs[0])
    _assert_same_results(rebuilt_outputs[0], rebuilt_outputs[1])
