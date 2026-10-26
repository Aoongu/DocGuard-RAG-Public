import hashlib
import json
import os
import socket
import sqlite3
from pathlib import Path, PurePosixPath

import pytest

from docguard.ingestion.models import SnapshotRecord
from docguard.ingestion.snapshot import load_snapshot_manifest, write_snapshot_manifest
from docguard.processing import pipeline as pipeline_module
from docguard.processing.chunker import chunk_document
from docguard.processing.config import load_chunking_config
from docguard.processing.html_parser import parse_postgresql_html
from docguard.processing.pipeline import (
    PipelineConflictError,
    PipelineError,
    build_corpus,
    main,
    verify_corpus,
)
from docguard.processing.storage import create_metadata_sqlite, logical_sqlite_digest
from docguard.processing.token_budget import TokenBudget

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "chunking" / "s2-chunk-v1.json"
TOKENIZERS = PROJECT_ROOT / "data" / "tokenizers" / "manifest.json"


def _record(version: str, payload: bytes) -> SnapshotRecord:
    major = version.split(".", 1)[0]
    return SnapshotRecord(
        schema_version=1,
        postgresql_version=version,
        topic="sql_commands",
        archive_path=PurePosixPath(
            f"data/archives/postgresql-{version}-docs.tar.gz"
        ),
        archive_sha256=("a" if version == "17.11" else "b") * 64,
        archive_member=PurePosixPath(
            f"postgresql-{version}/doc/src/sgml/html/sql-select.html"
        ),
        raw_path=PurePosixPath(f"data/raw/postgresql-{version}/sql-select.html"),
        canonical_url=f"https://www.postgresql.org/docs/{major}/sql-select.html",
        content_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_root = tmp_path / "raw"
    manifest = tmp_path / "snapshot.jsonl"
    output = tmp_path / "processed" / "s2-chunk-v1"
    records = []
    for version, marker in (("17.11", "seventeen"), ("18.6", "eighteen")):
        payload = (
            "<html><head><title>SELECT</title></head><body id='docContent'>"
            f"<div id='SQL-SELECT'><h2>SELECT</h2><p>{marker} rows.</p></div>"
            "</body></html>"
        ).encode()
        destination = raw_root / f"postgresql-{version}" / "sql-select.html"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(payload)
        records.append(_record(version, payload))
    write_snapshot_manifest(records, manifest)
    return raw_root, manifest, output


def _build(tmp_path: Path):
    raw_root, manifest, output = _fixture(tmp_path)
    result = build_corpus(
        raw_root,
        output,
        config_path=CONFIG,
        tokenizer_manifest_path=TOKENIZERS,
        snapshot_manifest_path=manifest,
    )
    return raw_root, manifest, output, result


def test_build_writes_stable_jsonl_sqlite_stats_and_build_info(tmp_path: Path) -> None:
    raw_root, manifest, output, result = _build(tmp_path)

    assert result.status == "built"
    assert result.document_count == 2
    assert result.chunk_count == 2
    assert {path.name for path in output.iterdir()} == {
        "build-info.json",
        "chunks.jsonl",
        "metadata.sqlite3",
        "stats.json",
    }
    chunk_bytes = (output / "chunks.jsonl").read_bytes()
    assert chunk_bytes.endswith(b"\n") and b"\r" not in chunk_bytes
    rows = [json.loads(line) for line in chunk_bytes.splitlines()]
    assert [row["postgresql_version"] for row in rows] == ["17.11", "18.6"]
    assert list(rows[0]) == [
        "schema_version",
        "chunk_id",
        "document_id",
        "postgresql_version",
        "topic",
        "canonical_url",
        "anchor",
        "section_path",
        "ordinal",
        "text",
        "token_count",
        "content_sha256",
        "chunking_version",
    ]
    build_info = json.loads((output / "build-info.json").read_text(encoding="utf-8"))
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert build_info["chunks_sha256"] == hashlib.sha256(chunk_bytes).hexdigest()
    assert build_info["sqlite_logical_sha256"] == logical_sqlite_digest(
        output / "metadata.sqlite3"
    )
    assert stats["document_count"] == 2
    assert stats["chunk_count"] == 2
    assert stats["empty_chunk_count"] == 0
    assert stats["over_limit_chunk_count"] == 0
    assert stats["cross_version_document_duplicate_rate"] == 0.0
    with sqlite3.connect(output / "metadata.sqlite3") as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2

    verified = verify_corpus(
        raw_root,
        output,
        config_path=CONFIG,
        tokenizer_manifest_path=TOKENIZERS,
        snapshot_manifest_path=manifest,
    )
    assert verified.status == "verified"


def test_second_identical_build_is_idempotently_skipped(tmp_path: Path) -> None:
    raw_root, manifest, output, _ = _build(tmp_path)

    second = build_corpus(
        raw_root,
        output,
        config_path=CONFIG,
        tokenizer_manifest_path=TOKENIZERS,
        snapshot_manifest_path=manifest,
    )

    assert second.status == "skipped"


def test_existing_conflict_is_not_overwritten(tmp_path: Path) -> None:
    raw_root, manifest, output, _ = _build(tmp_path)
    chunks = output / "chunks.jsonl"
    chunks.write_bytes(chunks.read_bytes() + b"tampered\n")
    before = chunks.read_bytes()

    with pytest.raises(PipelineConflictError, match="chunks.jsonl"):
        build_corpus(
            raw_root,
            output,
            config_path=CONFIG,
            tokenizer_manifest_path=TOKENIZERS,
            snapshot_manifest_path=manifest,
        )

    assert chunks.read_bytes() == before


def test_existing_hardlinked_output_file_is_rejected(tmp_path: Path) -> None:
    raw_root, manifest, output, _ = _build(tmp_path)
    chunks = output / "chunks.jsonl"
    os.link(chunks, tmp_path / "chunks-hardlink.jsonl")

    with pytest.raises(PipelineConflictError, match="unsafe output file"):
        verify_corpus(
            raw_root,
            output,
            config_path=CONFIG,
            tokenizer_manifest_path=TOKENIZERS,
            snapshot_manifest_path=manifest,
        )


def test_sqlite_transaction_rolls_back_duplicate_chunk_batch(tmp_path: Path) -> None:
    raw_root, manifest, _ = _fixture(tmp_path)
    record = load_snapshot_manifest(manifest)[0]
    document, blocks = parse_postgresql_html(
        (raw_root / "postgresql-17.11" / "sql-select.html").read_bytes(), record
    )
    config = load_chunking_config(CONFIG)
    budget = TokenBudget.from_manifest(TOKENIZERS, PROJECT_ROOT)
    chunks = chunk_document(document, blocks, config, budget)
    database = tmp_path / "failed.sqlite3"

    with pytest.raises(sqlite3.IntegrityError):
        create_metadata_sqlite(database, (document,), (*chunks, *chunks), {})

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0] == 0


def test_build_rejects_symlink_output_root(tmp_path: Path) -> None:
    raw_root, manifest, output = _fixture(tmp_path)
    target = tmp_path / "outside"
    target.mkdir()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(PipelineError, match="unsafe|symlink|output"):
        build_corpus(
            raw_root,
            output,
            config_path=CONFIG,
            tokenizer_manifest_path=TOKENIZERS,
            snapshot_manifest_path=manifest,
        )


def test_cli_build_verify_and_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw_root, manifest, output = _fixture(tmp_path)
    common = [
        "--raw-root",
        str(raw_root),
        "--output-root",
        str(output),
        "--config",
        str(CONFIG),
        "--tokenizer-manifest",
        str(TOKENIZERS),
        "--snapshot-manifest",
        str(manifest),
    ]

    assert main(["build", *common]) == 0
    assert main(["verify", *common]) == 0
    assert main(["stats", "--output-root", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert printed["document_count"] == 2


def test_build_is_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root, manifest, output = _fixture(tmp_path)

    def forbid_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    result = build_corpus(
        raw_root,
        output,
        config_path=CONFIG,
        tokenizer_manifest_path=TOKENIZERS,
        snapshot_manifest_path=manifest,
    )

    assert result.status == "built"


def test_failed_staging_is_retained_without_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, manifest, output = _fixture(tmp_path)
    original = pipeline_module.write_stage_file
    writes = 0

    def fail_second(stage: Path, name: str, data: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected staging failure")
        original(stage, name, data)

    monkeypatch.setattr(pipeline_module, "write_stage_file", fail_second)
    with pytest.raises(PipelineError, match="staging retained"):
        build_corpus(
            raw_root,
            output,
            config_path=CONFIG,
            tokenizer_manifest_path=TOKENIZERS,
            snapshot_manifest_path=manifest,
        )

    assert not output.exists()
    assert len(list(output.parent.glob(".s2-chunk-v1.stage-*"))) == 1


def test_invalid_raw_set_fails_before_publication_and_retains_stage_policy(
    tmp_path: Path,
) -> None:
    raw_root, manifest, output = _fixture(tmp_path)
    (raw_root / "postgresql-17.11" / "extra.html").write_text("extra")

    with pytest.raises(PipelineError, match="raw|file set"):
        build_corpus(
            raw_root,
            output,
            config_path=CONFIG,
            tokenizer_manifest_path=TOKENIZERS,
            snapshot_manifest_path=manifest,
        )

    assert not output.exists()

