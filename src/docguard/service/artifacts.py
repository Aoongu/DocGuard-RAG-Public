"""Read-only access to frozen corpus and evidence artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from docguard.fs_safety import FileSafetyError, validate_regular_file


class ArtifactUnavailableError(RuntimeError):
    """Raised when a required frozen artifact cannot be safely opened."""


class EvidenceNotFoundError(LookupError):
    """Raised when a chunk ID is absent from the frozen corpus."""


@dataclass(frozen=True, slots=True)
class CorpusArtifact:
    version: str
    document_count: int
    chunk_count: int
    index_status: str
    snapshot_sha256: str | None
    built_at: str | None


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    chunk_id: str
    postgresql_version: str
    topic: str
    section_path: tuple[str, ...]
    canonical_url: str
    anchor: str
    text: str
    content_sha256: str
    chunking_version: str


RebuildSubmitter = Callable[[], Awaitable[str]]


class ArtifactService:
    def __init__(
        self,
        metadata_db: Path,
        stats_path: Path,
        manifest_path: Path,
        *,
        rebuild_submitter: RebuildSubmitter | None = None,
    ) -> None:
        self._metadata_db = metadata_db
        self._stats_path = stats_path
        self._manifest_path = manifest_path
        self._rebuild_submitter = rebuild_submitter

    def _connect(self) -> sqlite3.Connection:
        try:
            validate_regular_file(self._metadata_db)
            return sqlite3.connect(f"file:{self._metadata_db.resolve().as_posix()}?mode=ro", uri=True)
        except (FileSafetyError, OSError, ValueError, sqlite3.Error) as error:
            raise ArtifactUnavailableError("metadata database is unavailable") from error

    def _identity(self) -> tuple[str | None, str | None]:
        try:
            validate_regular_file(self._stats_path)
            validate_regular_file(self._manifest_path)
            stats = json.loads(self._stats_path.read_text(encoding="utf-8"))
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (FileSafetyError, OSError, ValueError, json.JSONDecodeError) as error:
            raise ArtifactUnavailableError("corpus identity artifacts are unavailable") from error
        if not isinstance(stats, dict) or not isinstance(manifest, dict):
            raise ArtifactUnavailableError("corpus identity artifacts are invalid")
        digest = manifest.get("sha256")
        created_at = manifest.get("created_at")
        return (
            digest if isinstance(digest, str) else None,
            created_at if isinstance(created_at, str) else None,
        )

    def list_corpora(self) -> tuple[CorpusArtifact, ...]:
        snapshot_sha256, built_at = self._identity()
        connection = self._connect()
        try:
            documents = dict(
                connection.execute(
                    "SELECT postgresql_version, COUNT(*) FROM documents "
                    "GROUP BY postgresql_version ORDER BY postgresql_version"
                )
            )
            chunks = dict(
                connection.execute(
                    "SELECT postgresql_version, COUNT(*) FROM chunks "
                    "GROUP BY postgresql_version ORDER BY postgresql_version"
                )
            )
        except sqlite3.Error as error:
            raise ArtifactUnavailableError("metadata schema is unavailable") from error
        finally:
            connection.close()
        versions = sorted(set(documents) | set(chunks))
        return tuple(
            CorpusArtifact(
                version=version,
                document_count=int(documents.get(version, 0)),
                chunk_count=int(chunks.get(version, 0)),
                index_status="ready",
                snapshot_sha256=snapshot_sha256,
                built_at=built_at,
            )
            for version in versions
        )

    def get_evidence(self, evidence_id: str) -> EvidenceArtifact:
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT chunk_id, postgresql_version, topic, section_path_json, "
                "canonical_url, anchor, text, content_sha256, chunking_version "
                "FROM chunks WHERE chunk_id = ?",
                (evidence_id,),
            ).fetchone()
        except sqlite3.Error as error:
            raise ArtifactUnavailableError("metadata schema is unavailable") from error
        finally:
            connection.close()
        if row is None:
            raise EvidenceNotFoundError(f"evidence not found: {evidence_id}")
        section_path = json.loads(row[3])
        if not isinstance(section_path, list) or not all(
            isinstance(item, str) for item in section_path
        ):
            raise ArtifactUnavailableError("evidence section path is invalid")
        return EvidenceArtifact(
            chunk_id=row[0],
            postgresql_version=row[1],
            topic=row[2],
            section_path=tuple(section_path),
            canonical_url=row[4],
            anchor=row[5],
            text=row[6],
            content_sha256=row[7],
            chunking_version=row[8],
        )

    async def rebuild(self) -> str:
        if self._rebuild_submitter is None:
            raise ArtifactUnavailableError("corpus rebuild is not configured")
        return await self._rebuild_submitter()
