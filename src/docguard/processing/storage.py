"""Stable JSONL, SQLite metadata, and corpus statistics for S2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from docguard.processing.models import ChunkRecord, DocumentRecord

_SCHEMA = (
    (
        "CREATE TABLE documents (document_id TEXT PRIMARY KEY, postgresql_version TEXT NOT "
        "NULL, topic TEXT NOT NULL, canonical_url TEXT NOT NULL UNIQUE, title TEXT NOT NULL, "
        "content_sha256 TEXT NOT NULL)"
    ),
    (
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES "
        "documents(document_id), postgresql_version TEXT NOT NULL, topic TEXT NOT NULL, "
        "canonical_url TEXT NOT NULL, anchor TEXT NOT NULL, section_path_json TEXT NOT NULL, "
        "ordinal INTEGER NOT NULL CHECK (ordinal >= 0), text TEXT NOT NULL, token_count INTEGER "
        "NOT NULL CHECK (token_count > 0 AND token_count <= 384), content_sha256 TEXT NOT NULL, "
        "chunking_version TEXT NOT NULL, UNIQUE(document_id, anchor, ordinal))"
    ),
    "CREATE TABLE build_info (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)",
)


def chunk_dict(chunk: ChunkRecord) -> dict[str, object]:
    return {
        "schema_version": chunk.schema_version,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "postgresql_version": chunk.postgresql_version,
        "topic": chunk.topic,
        "canonical_url": chunk.canonical_url,
        "anchor": chunk.anchor,
        "section_path": list(chunk.section_path),
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "content_sha256": chunk.content_sha256,
        "chunking_version": chunk.chunking_version,
    }


def jsonl_bytes(chunks: Sequence[ChunkRecord]) -> bytes:
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            chunk.postgresql_version,
            chunk.canonical_url,
            chunk.anchor,
            chunk.ordinal,
        ),
    )
    return "".join(
        json.dumps(chunk_dict(chunk), ensure_ascii=False, separators=(",", ":")) + "\n"
        for chunk in ordered
    ).encode("utf-8")


def stable_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def create_metadata_sqlite(
    path: Path,
    documents: Sequence[DocumentRecord],
    chunks: Sequence[ChunkRecord],
    build_info: Mapping[str, object],
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA page_size = 4096")
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.document_id,
                        item.postgresql_version,
                        item.topic,
                        item.canonical_url,
                        item.title,
                        item.content_sha256,
                    )
                    for item in sorted(documents, key=lambda item: item.document_id)
                ],
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.chunk_id,
                        item.document_id,
                        item.postgresql_version,
                        item.topic,
                        item.canonical_url,
                        item.anchor,
                        json.dumps(list(item.section_path), ensure_ascii=False),
                        item.ordinal,
                        item.text,
                        item.token_count,
                        item.content_sha256,
                        item.chunking_version,
                    )
                    for item in sorted(chunks, key=lambda item: item.chunk_id)
                ],
            )
            connection.executemany(
                "INSERT INTO build_info VALUES (?, ?)",
                [
                    (
                        key,
                        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                    )
                    for key, value in sorted(build_info.items())
                ],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        connection.execute("VACUUM")
    finally:
        connection.close()


def metadata_sqlite_bytes(
    documents: Sequence[DocumentRecord],
    chunks: Sequence[ChunkRecord],
    build_info: Mapping[str, object],
) -> bytes:
    descriptor, name = tempfile.mkstemp(prefix="docguard-s2-", suffix=".sqlite3")
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    try:
        create_metadata_sqlite(path, documents, chunks, build_info)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def logical_sqlite_digest(path: Path) -> str:
    digest = hashlib.sha256()
    connection = sqlite3.connect(path)
    try:
        schemas = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE "
            "'sqlite_%' ORDER BY name"
        ).fetchall()
        for name, sql in schemas:
            digest.update(
                (json.dumps(["schema", name, sql], separators=(",", ":")) + "\n").encode()
            )
        queries = (
            ("build_info", "SELECT * FROM build_info ORDER BY key"),
            ("chunks", "SELECT * FROM chunks ORDER BY chunk_id"),
            ("documents", "SELECT * FROM documents ORDER BY document_id"),
        )
        for table, query in queries:
            for row in connection.execute(query):
                digest.update(
                    (
                        json.dumps([table, *row], ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                )
    finally:
        connection.close()
    return digest.hexdigest()


def logical_sqlite_digest_bytes(payload: bytes) -> str:
    descriptor, name = tempfile.mkstemp(prefix="docguard-s2-digest-", suffix=".sqlite3")
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
        return logical_sqlite_digest(path)
    finally:
        path.unlink(missing_ok=True)


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def corpus_stats(
    documents: Sequence[DocumentRecord], chunks: Sequence[ChunkRecord]
) -> dict[str, object]:
    token_counts = [chunk.token_count for chunk in chunks]
    by_version = Counter(chunk.postgresql_version for chunk in chunks)
    by_topic = Counter(chunk.topic for chunk in chunks)
    document_hashes: dict[str, list[str]] = defaultdict(list)
    hash_versions: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        document_hashes[document.postgresql_version].append(document.content_sha256)
        hash_versions[document.content_sha256].add(document.postgresql_version)
    within = {
        version: (len(values) - len(set(values))) / len(values) if values else 0.0
        for version, values in sorted(document_hashes.items())
    }
    cross = (
        sum(len(versions) > 1 for versions in hash_versions.values())
        / len(hash_versions)
        if hash_versions
        else 0.0
    )
    return {
        "schema_version": 1,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents_by_version": dict(
            sorted(Counter(item.postgresql_version for item in documents).items())
        ),
        "chunks_by_version": dict(sorted(by_version.items())),
        "chunks_by_topic": dict(sorted(by_topic.items())),
        "token_length": {
            "min": min(token_counts),
            "p50": _nearest_rank(token_counts, 0.50),
            "p95": _nearest_rank(token_counts, 0.95),
            "max": max(token_counts),
        },
        "within_version_document_duplicate_rate": within,
        "cross_version_document_duplicate_rate": cross,
        "empty_chunk_count": sum(not chunk.text for chunk in chunks),
        "over_limit_chunk_count": sum(chunk.token_count > 384 for chunk in chunks),
    }
