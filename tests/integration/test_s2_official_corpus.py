import hashlib
import json
import socket
from pathlib import Path

from docguard.ingestion.catalog import SOURCES
from docguard.ingestion.models import SNAPSHOT_TOPICS
from docguard.ingestion.snapshot import load_snapshot_manifest, restore_snapshot
from docguard.processing.pipeline import build_corpus, verify_corpus
from docguard.processing.storage import logical_sqlite_digest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "snapshot.jsonl"
S2_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "s2-chunk-v1.json"
CONFIG = PROJECT_ROOT / "configs" / "chunking" / "s2-chunk-v1.json"
TOKENIZERS = PROJECT_ROOT / "data" / "tokenizers" / "manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_official_corpus_build_is_offline_traceable_and_deterministic(
    tmp_path: Path, monkeypatch,
) -> None:
    committed = json.loads(S2_MANIFEST.read_text(encoding="utf-8"))

    def forbid_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("S2 official build must remain offline")

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    records = load_snapshot_manifest(SNAPSHOT_MANIFEST)
    raw_root = tmp_path / "raw"
    restore_snapshot(SOURCES, records, raw_root)
    first = tmp_path / "first" / "s2-chunk-v1"
    second = tmp_path / "second" / "s2-chunk-v1"
    first_result = build_corpus(
        raw_root,
        first,
        config_path=CONFIG,
        tokenizer_manifest_path=TOKENIZERS,
        snapshot_manifest_path=SNAPSHOT_MANIFEST,
    )
    verified = verify_corpus(
        raw_root,
        first,
        config_path=CONFIG,
        tokenizer_manifest_path=TOKENIZERS,
        snapshot_manifest_path=SNAPSHOT_MANIFEST,
    )
    second_result = build_corpus(
        raw_root,
        second,
        config_path=CONFIG,
        tokenizer_manifest_path=TOKENIZERS,
        snapshot_manifest_path=SNAPSHOT_MANIFEST,
    )

    assert first_result.document_count == verified.document_count == 773
    assert first_result.chunk_count == second_result.chunk_count
    first_info = json.loads((first / "build-info.json").read_text(encoding="utf-8"))
    second_info = json.loads((second / "build-info.json").read_text(encoding="utf-8"))
    assert first_info == second_info == committed
    assert _sha256(first / "chunks.jsonl") == _sha256(second / "chunks.jsonl")
    assert _sha256(first / "stats.json") == _sha256(second / "stats.json")
    assert logical_sqlite_digest(first / "metadata.sqlite3") == (
        logical_sqlite_digest(second / "metadata.sqlite3")
    )

    chunks = [
        json.loads(line)
        for line in (first / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(chunks) == first_result.chunk_count
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert {chunk["postgresql_version"] for chunk in chunks} == {"17.11", "18.6"}
    assert {chunk["topic"] for chunk in chunks} == SNAPSHOT_TOPICS
    assert all(chunk["text"] and 0 < chunk["token_count"] <= 384 for chunk in chunks)
    source_urls = {record.canonical_url for record in records}
    assert all(chunk["canonical_url"] in source_urls for chunk in chunks)
    assert all(
        chunk["canonical_url"] + "#" + chunk["anchor"]
        for chunk in chunks
    )

    stats = json.loads((first / "stats.json").read_text(encoding="utf-8"))
    assert stats["document_count"] == 773
    assert stats["chunk_count"] == len(chunks)
    assert stats["empty_chunk_count"] == 0
    assert stats["over_limit_chunk_count"] == 0
    assert stats["token_length"]["max"] <= 384
