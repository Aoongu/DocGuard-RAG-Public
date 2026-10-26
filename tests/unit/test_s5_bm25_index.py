from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from docguard.processing.models import ChunkRecord, make_chunk_id, make_document_id
from docguard.processing.storage import chunk_dict
from docguard.retrieval.bm25_index import (
    BM25IndexError,
    bm25_score,
    build_bm25_index,
    verify_bm25_index,
)


def _chunk(version: str, page: str, ordinal: int, text: str) -> ChunkRecord:
    major = version.split(".", 1)[0]
    url = f"https://www.postgresql.org/docs/{major}/{page}.html"
    anchor = f"section-{ordinal}"
    return ChunkRecord(
        schema_version=1,
        chunk_id=make_chunk_id("s2-chunk-v1", version, url, anchor, ordinal),
        document_id=make_document_id(version, url),
        postgresql_version=version,
        topic="installation_configuration",
        canonical_url=url,
        anchor=anchor,
        section_path=(page, f"Section {ordinal}"),
        ordinal=ordinal,
        text=text,
        token_count=len(text.split()),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        chunking_version="s2-chunk-v1",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[ChunkRecord]]:
    chunks = [
        _chunk("17.11", "runtime-config-wal", 0, "wal_level replica replica"),
        _chunk("17.11", "auth-pg-hba-conf", 0, "pg_hba.conf replica"),
        _chunk("18.6", "runtime-config-connection", 0, "--max-connections login"),
    ]
    payload = "".join(
        json.dumps(chunk_dict(item), ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in chunks
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
    return chunks_path, config_path, chunks


def test_bm25_formula_matches_hand_calculation() -> None:
    expected_idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected = expected_idf * (2 * 2.2) / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 2))

    assert bm25_score(2, 2, 3, 3, 2, k1=1.2, b=0.75) == pytest.approx(expected)


def test_build_persists_schema_counts_and_deterministic_logical_digest(
    tmp_path: Path,
) -> None:
    chunks_path, config_path, chunks = _fixture(tmp_path)
    first = build_bm25_index(chunks_path, tmp_path / "index-a", config_path=config_path)
    second = build_bm25_index(chunks_path, tmp_path / "index-b", config_path=config_path)

    assert first.status == second.status == "built"
    assert first.document_count == 3
    assert first.version_counts == {"17.11": 2, "18.6": 1}
    assert first.vocabulary_size > 0
    assert first.posting_count >= first.vocabulary_size
    assert first.logical_sha256 == second.logical_sha256
    assert verify_bm25_index(
        tmp_path / "index-a", chunks_path, config_path=config_path
    ).status == "verified"
    with sqlite3.connect(tmp_path / "index-a/bm25.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3
        replica = connection.execute(
            "SELECT document_frequency FROM terms WHERE term = 'replica'"
        ).fetchone()
        assert replica == (2,)
        ids = [row[0] for row in connection.execute("SELECT chunk_id FROM documents")]
        assert sorted(ids) == sorted(item.chunk_id for item in chunks)


def test_duplicate_chunk_id_fails_closed(tmp_path: Path) -> None:
    chunks_path, config_path, _ = _fixture(tmp_path)
    chunks_path.write_bytes(chunks_path.read_bytes() * 2)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chunks_sha256"] = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(BM25IndexError, match="duplicate chunk_id"):
        build_bm25_index(chunks_path, tmp_path / "index", config_path=config_path)


def test_verify_rejects_tampered_sqlite_metadata(tmp_path: Path) -> None:
    chunks_path, config_path, _ = _fixture(tmp_path)
    index_root = tmp_path / "index"
    build_bm25_index(chunks_path, index_root, config_path=config_path)
    with sqlite3.connect(index_root / "bm25.sqlite3") as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'tokenizer_version'",
            ('"tampered"',),
        )
        connection.commit()

    with pytest.raises(BM25IndexError, match="logical digest"):
        verify_bm25_index(index_root, chunks_path, config_path=config_path)


@pytest.mark.parametrize("mutation", ["hash", "version", "malformed"])
def test_invalid_frozen_input_fails_closed(tmp_path: Path, mutation: str) -> None:
    chunks_path, config_path, chunks = _fixture(tmp_path)
    if mutation == "hash":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["chunks_sha256"] = "0" * 64
        config_path.write_text(json.dumps(config), encoding="utf-8")
    elif mutation == "version":
        row = asdict(chunks[0])
        row["section_path"] = list(row["section_path"])
        row["chunking_version"] = "s2-chunk-v2"
        chunks_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["chunks_sha256"] = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        config_path.write_text(json.dumps(config), encoding="utf-8")
    else:
        chunks_path.write_text("{not-json}\n", encoding="utf-8")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["chunks_sha256"] = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(BM25IndexError):
        build_bm25_index(chunks_path, tmp_path / "index", config_path=config_path)

