from __future__ import annotations

import hashlib
import json
from pathlib import Path

from docguard.processing.models import ChunkRecord, make_chunk_id, make_document_id
from docguard.processing.storage import chunk_dict
from docguard.retrieval.hybrid_pipeline import main


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    text = "wal_level replica"
    url = "https://www.postgresql.org/docs/17/runtime-config-wal.html"
    chunk = ChunkRecord(
        schema_version=1,
        chunk_id=make_chunk_id("s2-chunk-v1", "17.11", url, "wal-level", 0),
        document_id=make_document_id("17.11", url),
        postgresql_version="17.11",
        topic="installation_configuration",
        canonical_url=url,
        anchor="wal-level",
        section_path=("WAL",),
        ordinal=0,
        text=text,
        token_count=2,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        chunking_version="s2-chunk-v1",
    )
    payload = (
        json.dumps(chunk_dict(chunk), ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_bytes(payload)
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/retrieval/s5-hybrid-v1.json").read_text(
            encoding="utf-8"
        )
    )
    config["chunks_sha256"] = hashlib.sha256(payload).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return chunks_path, config_path


def test_build_verify_and_stats_cli(tmp_path: Path, capsys) -> None:
    chunks, config = _fixture(tmp_path)
    output = tmp_path / "bm25"

    assert main(["build", "--chunks", str(chunks), "--config", str(config), "--output-root", str(output)]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["status"] == "built"
    assert main(["verify", "--chunks", str(chunks), "--config", str(config), "--index-root", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert main(["stats", "--index-root", str(output)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["document_count"] == 1
    assert stats["version_counts"] == {"17.11": 1}
    assert main([
        "search",
        "--index-root",
        str(output),
        "--config",
        str(config),
        "--query",
        "wal_level",
        "--version",
        "17.11",
    ]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["postgresql_version"] == "17.11"
    assert hits[0]["text"] == "wal_level replica"


def test_search_cli_accepts_query_that_starts_with_double_dash(
    tmp_path: Path, capsys
) -> None:
    chunks, config = _fixture(tmp_path)
    output = tmp_path / "bm25"
    assert main([
        "build",
        "--chunks",
        str(chunks),
        "--config",
        str(config),
        "--output-root",
        str(output),
    ]) == 0
    capsys.readouterr()

    assert main([
        "search",
        "--index-root",
        str(output),
        "--config",
        str(config),
        "--query=--max-connections",
        "--version",
        "17.11",
    ]) == 0
    assert json.loads(capsys.readouterr().out) == []
