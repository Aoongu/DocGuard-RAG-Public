"""Deterministic SQLite BM25 index for the frozen S2 corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from docguard.fs_safety import (
    FileConflictError,
    FileSafetyError,
    atomic_rename_directory_noreplace,
    create_stage,
    ensure_directory,
    exclusive_lock,
    validate_directory,
    validate_regular_file,
    verify_stage,
    write_stage_file,
)
from docguard.processing.models import ChunkRecord
from docguard.processing.storage import stable_json_bytes
from docguard.retrieval.hybrid_config import HybridConfig, load_hybrid_config
from docguard.retrieval.technical_tokenizer import tokenize_technical

_OUTPUT_FILES = frozenset({"bm25.sqlite3", "manifest.json"})
_SCHEMA = (
    "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    (
        "CREATE TABLE documents(chunk_id TEXT PRIMARY KEY, postgresql_version TEXT NOT NULL, "
        "topic TEXT NOT NULL, canonical_url TEXT NOT NULL, anchor TEXT NOT NULL, "
        "section_path_json TEXT NOT NULL, body TEXT NOT NULL, doc_len INTEGER NOT NULL)"
    ),
    "CREATE TABLE terms(term TEXT PRIMARY KEY, document_frequency INTEGER NOT NULL)",
    (
        "CREATE TABLE postings(term TEXT NOT NULL, chunk_id TEXT NOT NULL, "
        "term_frequency INTEGER NOT NULL, PRIMARY KEY(term, chunk_id))"
    ),
    "CREATE INDEX postings_chunk_idx ON postings(chunk_id)",
    "CREATE INDEX documents_version_idx ON documents(postgresql_version)",
)


class BM25IndexError(RuntimeError):
    """Raised when an S5 BM25 index violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class BM25BuildInfo:
    status: str
    document_count: int
    vocabulary_size: int
    posting_count: int
    version_counts: dict[str, int]
    logical_sha256: str


@dataclass(frozen=True, slots=True)
class BM25SearchHit:
    rank: int
    chunk_id: str
    score: float
    postgresql_version: str
    topic: str
    section_path: tuple[str, ...]
    canonical_url: str
    anchor: str
    text: str


def bm25_score(
    term_frequency: int,
    document_frequency: int,
    document_length: int,
    document_count: int,
    average_document_length: float,
    *,
    k1: float,
    b: float,
) -> float:
    values = (
        term_frequency,
        document_frequency,
        document_length,
        document_count,
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("BM25 counts and lengths must be positive integers")
    if document_frequency > document_count:
        raise ValueError("document_frequency cannot exceed document_count")
    if not math.isfinite(average_document_length) or average_document_length <= 0:
        raise ValueError("average_document_length must be positive and finite")
    if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("invalid BM25 constants")
    inverse_document_frequency = math.log(
        1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    denominator = term_frequency + k1 * (
        1 - b + b * document_length / average_document_length
    )
    score = inverse_document_frequency * term_frequency * (k1 + 1) / denominator
    if not math.isfinite(score):
        raise BM25IndexError("BM25 score must be finite")
    return score


def _sha256(path: Path) -> str:
    try:
        validate_regular_file(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, FileSafetyError) as error:
        raise BM25IndexError(f"unable to hash input file: {path}") from error


def _load_chunks(path: Path, config: HybridConfig) -> list[ChunkRecord]:
    if _sha256(path) != config.chunks_sha256:
        raise BM25IndexError("chunks SHA-256 does not match the S5 config")
    records: list[ChunkRecord] = []
    seen: set[str] = set()
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError("chunk record must be an object")
            if isinstance(data.get("section_path"), list):
                data["section_path"] = tuple(data["section_path"])
            record = ChunkRecord(**data)
            if record.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id at line {line_number}")
            if record.chunking_version != config.chunking_version:
                raise ValueError(f"unexpected chunking_version at line {line_number}")
            if not tokenize_technical(record.text):
                raise ValueError(f"empty technical token stream at line {line_number}")
            seen.add(record.chunk_id)
            records.append(record)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise BM25IndexError(f"invalid chunks JSONL: {path}: {error}") from error
    if not records:
        raise BM25IndexError("chunks JSONL must not be empty")
    return records


def _logical_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    schemas = connection.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "AND type IN ('table', 'index') ORDER BY type, name"
    ).fetchall()
    for row in schemas:
        digest.update((json.dumps(["schema", *row], separators=(",", ":")) + "\n").encode())
    queries = (
        ("metadata", "SELECT key, value FROM metadata ORDER BY key"),
        ("documents", "SELECT * FROM documents ORDER BY chunk_id"),
        ("terms", "SELECT * FROM terms ORDER BY term"),
        ("postings", "SELECT * FROM postings ORDER BY term, chunk_id"),
    )
    for table, query in queries:
        for row in connection.execute(query):
            digest.update(
                (
                    json.dumps([table, *row], ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode()
            )
    return digest.hexdigest()


def logical_bm25_digest(path: Path) -> str:
    try:
        validate_regular_file(path)
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            return _logical_digest(connection)
        finally:
            connection.close()
    except (OSError, sqlite3.Error, FileSafetyError) as error:
        raise BM25IndexError(f"unable to inspect BM25 SQLite index: {path}") from error


def _base_metadata(
    config: HybridConfig, config_path: Path, chunks: list[ChunkRecord]
) -> dict[str, object]:
    version_counts = dict(sorted(Counter(item.postgresql_version for item in chunks).items()))
    return {
        "schema_version": 1,
        "retrieval_version": config.retrieval_version,
        "tokenizer_version": config.tokenizer_version,
        "chunking_version": config.chunking_version,
        "config_sha256": _sha256(config_path),
        "chunks_sha256": config.chunks_sha256,
        "k1": config.k1,
        "b": config.b,
        "document_count": len(chunks),
        "version_counts": version_counts,
    }


def _sqlite_payload(
    chunks: list[ChunkRecord], metadata: dict[str, object]
) -> tuple[bytes, int, int, str]:
    descriptor, name = tempfile.mkstemp(prefix="docguard-s5-", suffix=".sqlite3")
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    try:
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA page_size = 4096")
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA:
                connection.execute(statement)
            term_documents: Counter[str] = Counter()
            postings: list[tuple[str, str, int]] = []
            documents: list[tuple[object, ...]] = []
            for chunk in sorted(chunks, key=lambda item: item.chunk_id):
                frequencies = Counter(tokenize_technical(chunk.text))
                documents.append(
                    (
                        chunk.chunk_id,
                        chunk.postgresql_version,
                        chunk.topic,
                        chunk.canonical_url,
                        chunk.anchor,
                        json.dumps(list(chunk.section_path), ensure_ascii=False, separators=(",", ":")),
                        chunk.text,
                        sum(frequencies.values()),
                    )
                )
                for term, frequency in sorted(frequencies.items()):
                    postings.append((term, chunk.chunk_id, frequency))
                    term_documents[term] += 1
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [
                    (key, json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
                    for key, value in sorted(metadata.items())
                ],
            )
            connection.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)", documents)
            connection.executemany(
                "INSERT INTO terms VALUES (?, ?)", sorted(term_documents.items())
            )
            connection.executemany(
                "INSERT INTO postings VALUES (?, ?, ?)", sorted(postings)
            )
            connection.commit()
            logical = _logical_digest(connection)
            connection.execute("VACUUM")
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return path.read_bytes(), len(term_documents), len(postings), logical
    except sqlite3.Error as error:
        raise BM25IndexError("unable to create BM25 SQLite index") from error
    finally:
        path.unlink(missing_ok=True)


def _artifacts(
    chunks_path: Path, config_path: Path
) -> tuple[dict[str, bytes], BM25BuildInfo]:
    try:
        config = load_hybrid_config(config_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise BM25IndexError(f"invalid S5 config: {config_path}") from error
    chunks = _load_chunks(chunks_path, config)
    metadata = _base_metadata(config, config_path, chunks)
    sqlite_payload, vocabulary_size, posting_count, logical = _sqlite_payload(
        chunks, metadata
    )
    version_counts = dict(metadata["version_counts"])
    manifest = {
        **metadata,
        "vocabulary_size": vocabulary_size,
        "posting_count": posting_count,
        "logical_sha256": logical,
    }
    info = BM25BuildInfo(
        status="built",
        document_count=len(chunks),
        vocabulary_size=vocabulary_size,
        posting_count=posting_count,
        version_counts=version_counts,
        logical_sha256=logical,
    )
    return {
        "bm25.sqlite3": sqlite_payload,
        "manifest.json": stable_json_bytes(manifest),
    }, info


def _verify_files(index_root: Path, expected: dict[str, bytes]) -> None:
    try:
        validate_directory(index_root)
        children = {path.name: path for path in index_root.iterdir()}
        if set(children) != _OUTPUT_FILES:
            raise BM25IndexError("BM25 index output file set mismatch")
        for child in children.values():
            validate_regular_file(child)
        if children["manifest.json"].read_bytes() != expected["manifest.json"]:
            raise BM25IndexError("BM25 manifest differs from frozen inputs")
        expected_digest = logical_bm25_digest_bytes(expected["bm25.sqlite3"])
        if logical_bm25_digest(children["bm25.sqlite3"]) != expected_digest:
            raise BM25IndexError("BM25 logical digest mismatch")
    except BM25IndexError:
        raise
    except (OSError, FileSafetyError) as error:
        raise BM25IndexError(f"unsafe BM25 index: {index_root}") from error


def logical_bm25_digest_bytes(payload: bytes) -> str:
    descriptor, name = tempfile.mkstemp(prefix="docguard-s5-verify-", suffix=".sqlite3")
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
        return logical_bm25_digest(Path(name))
    finally:
        Path(name).unlink(missing_ok=True)


def build_bm25_index(
    chunks_path: Path,
    output_root: Path,
    *,
    config_path: Path = Path("configs/retrieval/s5-hybrid-v1.json"),
) -> BM25BuildInfo:
    expected, info = _artifacts(chunks_path, config_path)
    parent = output_root.parent
    try:
        ensure_directory(parent)
        with exclusive_lock(parent, output_root.name + ".build"):
            if os.path.lexists(output_root):
                _verify_files(output_root, expected)
                return replace(info, status="skipped")
            stage, identity = create_stage(parent, f".{output_root.name}.stage-")
            try:
                for name, payload in sorted(expected.items()):
                    write_stage_file(stage, name, payload)
                verify_stage(stage, identity, expected)
                atomic_rename_directory_noreplace(stage, output_root)
            except Exception as error:
                raise BM25IndexError(
                    f"staging retained for manual inspection: {stage}: {error}"
                ) from error
    except FileConflictError as error:
        raise BM25IndexError(str(error)) from error
    except FileSafetyError as error:
        raise BM25IndexError(str(error)) from error
    return info


def verify_bm25_index(
    index_root: Path,
    chunks_path: Path,
    *,
    config_path: Path = Path("configs/retrieval/s5-hybrid-v1.json"),
) -> BM25BuildInfo:
    expected, info = _artifacts(chunks_path, config_path)
    _verify_files(index_root, expected)
    return replace(info, status="verified")


def bm25_index_stats(index_root: Path) -> BM25BuildInfo:
    try:
        validate_directory(index_root)
        manifest_path = index_root / "manifest.json"
        validate_regular_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "document_count",
            "vocabulary_size",
            "posting_count",
            "version_counts",
            "logical_sha256",
        }
        if not isinstance(manifest, dict) or not required <= set(manifest):
            raise ValueError("manifest fields are incomplete")
        actual = logical_bm25_digest(index_root / "bm25.sqlite3")
        if actual != manifest["logical_sha256"]:
            raise ValueError("logical digest mismatch")
        return BM25BuildInfo(
            status="stats",
            document_count=int(manifest["document_count"]),
            vocabulary_size=int(manifest["vocabulary_size"]),
            posting_count=int(manifest["posting_count"]),
            version_counts=dict(manifest["version_counts"]),
            logical_sha256=str(manifest["logical_sha256"]),
        )
    except (OSError, FileSafetyError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise BM25IndexError(f"invalid BM25 index: {index_root}") from error


class BM25Retriever:
    """Lazy read-only BM25 retriever with version-scoped corpus statistics."""

    def __init__(
        self,
        index_root: Path,
        *,
        config_path: Path = Path("configs/retrieval/s5-hybrid-v1.json"),
    ) -> None:
        self._index_root = index_root
        self._config_path = config_path
        self._config = load_hybrid_config(config_path)
        self._connection: sqlite3.Connection | None = None

    def _open(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        try:
            validate_directory(self._index_root)
            manifest_path = self._index_root / "manifest.json"
            database_path = self._index_root / "bm25.sqlite3"
            validate_regular_file(manifest_path)
            validate_regular_file(database_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "retrieval_version": self._config.retrieval_version,
                "tokenizer_version": self._config.tokenizer_version,
                "chunking_version": self._config.chunking_version,
                "config_sha256": _sha256(self._config_path),
                "chunks_sha256": self._config.chunks_sha256,
                "k1": self._config.k1,
                "b": self._config.b,
            }
            if not isinstance(manifest, dict) or any(
                manifest.get(key) != value for key, value in expected.items()
            ):
                raise BM25IndexError("BM25 manifest does not match the active config")
            if logical_bm25_digest(database_path) != manifest.get("logical_sha256"):
                raise BM25IndexError("BM25 logical digest mismatch")
            connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            self._connection = connection
            return connection
        except BM25IndexError:
            raise
        except (OSError, sqlite3.Error, FileSafetyError, json.JSONDecodeError) as error:
            raise BM25IndexError(f"unable to open BM25 index: {self._index_root}") from error

    async def search(
        self,
        query: str,
        version: str | None,
        top_k: int,
        bm25_enabled: bool = True,
    ) -> list[BM25SearchHit]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if version is not None and version not in {"17.11", "18.6"}:
            raise ValueError("version must be null, 17.11, or 18.6")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if type(bm25_enabled) is not bool:
            raise TypeError("bm25_enabled must be a boolean")
        if not bm25_enabled:
            return []
        terms = tuple(sorted(set(tokenize_technical(query))))
        if not terms:
            raise ValueError("query must contain at least one technical token")

        connection = self._open()
        version_clause = "" if version is None else " WHERE postgresql_version = ?"
        version_parameters: tuple[str, ...] = () if version is None else (version,)
        stats = connection.execute(
            "SELECT COUNT(*), AVG(doc_len) FROM documents" + version_clause,
            version_parameters,
        ).fetchone()
        document_count = int(stats[0])
        if document_count == 0 or stats[1] is None:
            return []
        average_document_length = float(stats[1])

        placeholders = ",".join("?" for _ in terms)
        df_sql = (
            "SELECT p.term, COUNT(DISTINCT p.chunk_id) FROM postings p "
            "JOIN documents d ON d.chunk_id = p.chunk_id "
            f"WHERE p.term IN ({placeholders})"
        )
        candidate_sql = (
            "SELECT p.term, p.term_frequency, d.chunk_id, d.postgresql_version, d.topic, "
            "d.canonical_url, d.anchor, d.section_path_json, d.body, d.doc_len "
            "FROM postings p JOIN documents d ON d.chunk_id = p.chunk_id "
            f"WHERE p.term IN ({placeholders})"
        )
        parameters: tuple[object, ...] = terms
        if version is not None:
            df_sql += " AND d.postgresql_version = ?"
            candidate_sql += " AND d.postgresql_version = ?"
            parameters += (version,)
        df_sql += " GROUP BY p.term ORDER BY p.term"
        candidate_sql += " ORDER BY d.chunk_id, p.term"
        document_frequencies = {
            str(row[0]): int(row[1])
            for row in connection.execute(df_sql, parameters).fetchall()
        }
        scores: dict[str, float] = {}
        metadata: dict[str, sqlite3.Row] = {}
        for row in connection.execute(candidate_sql, parameters):
            chunk_id = str(row[2])
            if chunk_id in metadata and tuple(metadata[chunk_id][2:]) != tuple(row[2:]):
                raise BM25IndexError("duplicate chunk metadata is inconsistent")
            metadata[chunk_id] = row
            score = bm25_score(
                int(row[1]),
                document_frequencies[str(row[0])],
                int(row[9]),
                document_count,
                average_document_length,
                k1=self._config.k1,
                b=self._config.b,
            )
            scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        hits: list[BM25SearchHit] = []
        for rank, (chunk_id, score) in enumerate(ordered, start=1):
            if not math.isfinite(score):
                raise BM25IndexError("BM25 aggregate score must be finite")
            row = metadata[chunk_id]
            try:
                section_path = json.loads(row[7])
            except (TypeError, json.JSONDecodeError) as error:
                raise BM25IndexError("BM25 section_path metadata is invalid") from error
            if not isinstance(section_path, list) or not section_path or not all(
                isinstance(item, str) and item for item in section_path
            ):
                raise BM25IndexError("BM25 section_path metadata is invalid")
            hits.append(
                BM25SearchHit(
                    rank=rank,
                    chunk_id=chunk_id,
                    score=score,
                    postgresql_version=str(row[3]),
                    topic=str(row[4]),
                    canonical_url=str(row[5]),
                    anchor=str(row[6]),
                    section_path=tuple(section_path),
                    text=str(row[8]),
                )
            )
        return hits

    async def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
