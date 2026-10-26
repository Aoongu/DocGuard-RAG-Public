from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from docguard.service.artifacts import (
    ArtifactService,
    ArtifactUnavailableError,
    EvidenceNotFoundError,
)


@pytest.fixture
def metadata_db(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY, postgresql_version TEXT, topic TEXT,
            canonical_url TEXT, title TEXT, content_sha256 TEXT
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, document_id TEXT, postgresql_version TEXT,
            topic TEXT, canonical_url TEXT, anchor TEXT, section_path_json TEXT,
            ordinal INTEGER, text TEXT, token_count INTEGER, content_sha256 TEXT,
            chunking_version TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("d17", "17.11", "admin", "https://example/17", "Admin", "a" * 64),
            ("d18", "18.6", "admin", "https://example/18", "Admin", "b" * 64),
        ],
    )
    connection.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "chk_17_a",
                "d17",
                "17.11",
                "admin",
                "https://example/17",
                "vacuum",
                json.dumps(["Reference", "VACUUM"]),
                0,
                "VACUUM reclaims storage.",
                4,
                "c" * 64,
                "s2-chunk-v1",
            ),
            (
                "chk_17_b",
                "d17",
                "17.11",
                "admin",
                "https://example/17",
                "analyze",
                json.dumps(["Reference", "ANALYZE"]),
                0,
                "ANALYZE updates statistics.",
                4,
                "d" * 64,
                "s2-chunk-v1",
            ),
            (
                "chk_18_a",
                "d18",
                "18.6",
                "admin",
                "https://example/18",
                "vacuum",
                json.dumps(["Reference", "VACUUM"]),
                0,
                "VACUUM reclaims storage.",
                4,
                "e" * 64,
                "s2-chunk-v1",
            ),
        ],
    )
    connection.commit()
    connection.close()
    return path


def _stats(tmp_path: Path) -> Path:
    path = tmp_path / "stats.json"
    path.write_text(
        json.dumps(
            {
                "documents_by_version": {"17.11": 1, "18.6": 1},
                "chunks_by_version": {"17.11": 2, "18.6": 1},
            }
        ),
        encoding="utf-8",
    )
    return path


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"sha256": "f" * 64, "created_at": "2026-10-20T00:00:00Z"}),
        encoding="utf-8",
    )
    return path


def test_corpora_groups_documents_and_chunks_by_version(
    tmp_path: Path, metadata_db: Path
) -> None:
    rows = ArtifactService(metadata_db, _stats(tmp_path), _manifest(tmp_path)).list_corpora()

    assert [(row.version, row.document_count, row.chunk_count) for row in rows] == [
        ("17.11", 1, 2),
        ("18.6", 1, 1),
    ]
    assert all(row.index_status == "ready" for row in rows)


def test_evidence_lookup_returns_traceable_chunk(
    tmp_path: Path, metadata_db: Path
) -> None:
    evidence = ArtifactService(
        metadata_db, _stats(tmp_path), _manifest(tmp_path)
    ).get_evidence("chk_17_a")

    assert evidence.chunk_id == "chk_17_a"
    assert evidence.section_path == ("Reference", "VACUUM")
    assert evidence.canonical_url == "https://example/17"


def test_evidence_lookup_rejects_unknown_id(tmp_path: Path, metadata_db: Path) -> None:
    service = ArtifactService(metadata_db, _stats(tmp_path), _manifest(tmp_path))

    with pytest.raises(EvidenceNotFoundError, match="not found"):
        service.get_evidence("chk_missing")


def test_missing_metadata_is_dependency_failure(tmp_path: Path) -> None:
    service = ArtifactService(
        tmp_path / "missing.sqlite3", _stats(tmp_path), _manifest(tmp_path)
    )

    with pytest.raises(ArtifactUnavailableError, match="metadata"):
        service.list_corpora()


@pytest.mark.asyncio
async def test_rebuild_is_delegated_without_running_algorithm(
    tmp_path: Path, metadata_db: Path
) -> None:
    calls: list[str] = []

    async def submit() -> str:
        calls.append("submitted")
        return "rebuild-1"

    service = ArtifactService(
        metadata_db, _stats(tmp_path), _manifest(tmp_path), rebuild_submitter=submit
    )

    assert await service.rebuild() == "rebuild-1"
    assert calls == ["submitted"]
