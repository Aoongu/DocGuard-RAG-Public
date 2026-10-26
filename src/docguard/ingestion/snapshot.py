"""Deterministic, fail-closed PostgreSQL documentation snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from docguard.fs_safety import (
    FileConflictError,
    FileSafetyError,
    atomic_rename_directory_noreplace,
    ensure_directory,
    exclusive_lock,
    validate_directory,
    write_stage_file,
)
from docguard.ingestion.catalog import SOURCES, TOPIC_RULES, assign_topics
from docguard.ingestion.download import (
    DownloadError,
    download_archive,
)
from docguard.ingestion.models import ArchiveSource, SnapshotRecord

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    max_selected_members: int = 2_000
    max_member_bytes: int = 8 * 1024 * 1024
    max_total_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_selected_members", "max_member_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class RestoreResult:
    restored: int
    skipped: int


class SnapshotError(RuntimeError):
    """Raised when a snapshot cannot be validated or restored."""


class UnsafeArchiveError(SnapshotError):
    """Raised when a tar archive violates the extraction policy."""


class SnapshotConflictError(SnapshotError):
    """Raised when existing output differs from the frozen snapshot."""


def _sha256_path(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise SnapshotError(f"unable to read file: {path}") from error
    return size, digest.hexdigest()


def _validate_member_path(name: str) -> PurePosixPath:
    if (
        not name
        or "\\" in name
        or PureWindowsPath(name).drive
        or any(character in name for character in ("\x00", "\r", "\n", "\t"))
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise UnsafeArchiveError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise UnsafeArchiveError(f"unsafe archive member path: {name!r}")
    return path


def _read_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    limits: ExtractionLimits,
    total_read: int,
) -> tuple[bytes, int]:
    if member.size < 0 or member.size > limits.max_member_bytes:
        raise UnsafeArchiveError(f"archive member size limit exceeded: {member.name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise UnsafeArchiveError(f"unable to read regular member: {member.name}")
    data = bytearray()
    try:
        with stream:
            while True:
                remaining_member = limits.max_member_bytes - len(data)
                remaining_total = limits.max_total_bytes - total_read - len(data)
                if remaining_member < 0 or remaining_total < 0:
                    raise UnsafeArchiveError("archive actual-read limit exceeded")
                chunk = stream.read(min(_CHUNK_SIZE, remaining_member + 1, remaining_total + 1))
                if not chunk:
                    break
                data.extend(chunk)
                if (
                    len(data) > limits.max_member_bytes
                    or total_read + len(data) > limits.max_total_bytes
                    or len(data) > member.size
                ):
                    raise UnsafeArchiveError("archive actual-read limit exceeded")
    except (OSError, EOFError, tarfile.TarError) as error:
        raise UnsafeArchiveError(f"unable to read archive member: {member.name}") from error
    if len(data) != member.size:
        raise UnsafeArchiveError(f"archive member declared-size mismatch: {member.name}")
    return bytes(data), total_read + len(data)


def inspect_archive(
    archive_path: Path,
    version: str,
    *,
    limits: ExtractionLimits = ExtractionLimits(),  # noqa: B008
) -> dict[PurePosixPath, bytes]:
    """Validate every tar member and return selected HTML bytes by member path."""
    selected: dict[PurePosixPath, bytes] = {}
    total_read = 0
    selected_count = 0
    expected_root = PurePosixPath(f"postgresql-{version}/doc/src/sgml/html")
    saw_expected_root = False
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if member.isdir():
                    path = _validate_member_path(member.name.rstrip("/"))
                    saw_expected_root = saw_expected_root or path == expected_root
                    continue
                if not member.isfile():
                    raise UnsafeArchiveError(
                        f"archive contains a non-regular member: {member.name}"
                    )
                path = _validate_member_path(member.name)
                if path.parent != expected_root:
                    continue
                saw_expected_root = True
                if not path.name.endswith(".html") or not assign_topics((path.name,)):
                    continue
                selected_count += 1
                if selected_count > limits.max_selected_members:
                    raise UnsafeArchiveError("archive selected-member limit exceeded")
                if path in selected:
                    raise UnsafeArchiveError(f"duplicate archive member: {member.name}")
                data, total_read = _read_member(
                    archive, member, limits, total_read
                )
                selected[path] = data
    except UnsafeArchiveError:
        raise
    except (OSError, EOFError, tarfile.TarError) as error:
        raise UnsafeArchiveError(f"unable to inspect archive: {archive_path}") from error
    if not saw_expected_root:
        raise UnsafeArchiveError(
            f"archive does not contain expected HTML root: {expected_root}"
        )
    return selected


def _verify_archive_file(source: ArchiveSource, path: Path) -> None:
    size, digest = _sha256_path(path)
    if size != source.size_bytes:
        raise SnapshotError(f"archive size mismatch: {path}")
    if digest != source.sha256:
        raise SnapshotError(f"archive hash mismatch: {path}")


def build_snapshot_records(
    source: ArchiveSource,
    archive: Path,
    *,
    limits: ExtractionLimits = ExtractionLimits(),  # noqa: B008
) -> tuple[SnapshotRecord, ...]:
    """Validate an archive and build stable records for whitelisted HTML."""
    _verify_archive_file(source, archive)
    members = inspect_archive(archive, source.version, limits=limits)
    topics = assign_topics(path.name for path in members)
    records = []
    for member, data in members.items():
        topic = topics.get(member.name)
        if topic is None:
            continue
        records.append(
            SnapshotRecord(
                schema_version=1,
                postgresql_version=source.version,
                topic=topic,
                archive_path=source.archive_path,
                archive_sha256=source.sha256,
                archive_member=member,
                raw_path=PurePosixPath(
                    "data", "raw", f"postgresql-{source.version}", member.name
                ),
                canonical_url=(
                    f"https://www.postgresql.org/docs/{source.major_version}/{member.name}"
                ),
                content_sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )
        )
    return tuple(sorted(records, key=_record_sort_key))


def _source_dict(source: ArchiveSource) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "version": source.version,
        "major_version": source.major_version,
        "url": source.url,
        "archive_path": str(source.archive_path),
        "size_bytes": source.size_bytes,
        "sha256": source.sha256,
        "retrieved_at": source.retrieved_at,
        "license_name": source.license_name,
        "license_url": source.license_url,
        "copyright": "PostgreSQL Global Development Group",
        "allowed_topics": [rule.name for rule in TOPIC_RULES],
        "selection_rules": [
            {"topic": rule.name, "patterns": list(rule.patterns)}
            for rule in TOPIC_RULES
        ],
    }


def _record_dict(record: SnapshotRecord) -> dict[str, Any]:
    payload = asdict(record)
    for field in ("archive_path", "archive_member", "raw_path"):
        payload[field] = str(payload[field])
    return payload


def _atomic_text(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_sources_manifest(
    sources: Sequence[ArchiveSource], destination: Path
) -> None:
    ordered = sorted(sources, key=lambda item: item.version)
    if len({item.version for item in ordered}) != len(ordered):
        raise SnapshotError("duplicate source version")
    text = json.dumps(
        [_source_dict(source) for source in ordered],
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    _atomic_text(destination, text)


def _record_sort_key(record: SnapshotRecord) -> tuple[str, str, str]:
    return record.postgresql_version, record.topic, str(record.raw_path)


def _validate_records(records: Sequence[SnapshotRecord]) -> tuple[SnapshotRecord, ...]:
    unique: dict[str, set[object]] = {
        "archive_member": set(),
        "raw_path": set(),
        "canonical_url": set(),
    }
    ordered = tuple(sorted(records, key=_record_sort_key))
    for record in ordered:
        keys = {
            "archive_member": (record.postgresql_version, record.archive_member),
            "raw_path": record.raw_path,
            "canonical_url": record.canonical_url,
        }
        for label, key in keys.items():
            if key in unique[label]:
                raise SnapshotError(f"duplicate manifest {label}: {key}")
            unique[label].add(key)
    return ordered


def write_snapshot_manifest(
    records: Sequence[SnapshotRecord], destination: Path
) -> None:
    ordered = _validate_records(records)
    text = "".join(
        json.dumps(_record_dict(record), ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for record in ordered
    )
    _atomic_text(destination, text)


def _record_from_dict(payload: object, line_number: int) -> SnapshotRecord:
    if not isinstance(payload, dict):
        raise SnapshotError(f"manifest line {line_number} must be an object")
    converted = dict(payload)
    for field in ("archive_path", "archive_member", "raw_path"):
        if field in converted and isinstance(converted[field], str):
            converted[field] = PurePosixPath(converted[field])
    try:
        return SnapshotRecord(**converted)
    except (TypeError, ValueError) as error:
        raise SnapshotError(f"invalid manifest line {line_number}: {error}") from error


def load_snapshot_manifest(path: Path) -> tuple[SnapshotRecord, ...]:
    records = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise SnapshotError(f"blank manifest line {line_number}")
                try:
                    payload = json.loads(line, object_pairs_hook=_unique_json_object)
                except json.JSONDecodeError as error:
                    raise SnapshotError(
                        f"invalid JSON on manifest line {line_number}"
                    ) from error
                records.append(_record_from_dict(payload, line_number))
    except OSError as error:
        raise SnapshotError(f"unable to read snapshot manifest: {path}") from error
    return _validate_records(records)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _source_map(sources: Sequence[ArchiveSource]) -> dict[str, ArchiveSource]:
    result = {source.version: source for source in sources}
    if len(result) != len(sources):
        raise SnapshotError("duplicate source version")
    return result


def _validate_record_sources(
    sources: Sequence[ArchiveSource], records: Sequence[SnapshotRecord]
) -> dict[str, ArchiveSource]:
    source_by_version = _source_map(sources)
    _validate_records(records)
    record_versions = {record.postgresql_version for record in records}
    if record_versions != set(source_by_version):
        missing = sorted(set(source_by_version) - record_versions)
        extra = sorted(record_versions - set(source_by_version))
        raise SnapshotError(
            f"manifest/source version mismatch; missing={missing}, extra={extra}"
        )
    for record in records:
        source = source_by_version.get(record.postgresql_version)
        if source is None:
            raise SnapshotError(f"missing source for version {record.postgresql_version}")
        if (
            record.archive_path != source.archive_path
            or record.archive_sha256 != source.sha256
        ):
            raise SnapshotError("manifest record does not match its archive source")
    return source_by_version


def _selected_bytes(
    source: ArchiveSource,
    records: Sequence[SnapshotRecord],
    limits: ExtractionLimits,
) -> dict[str, bytes]:
    archive_path = Path(source.archive_path)
    _verify_archive_file(source, archive_path)
    available = inspect_archive(archive_path, source.version, limits=limits)
    assigned = assign_topics(path.name for path in available)
    manifest_members = {record.archive_member for record in records}
    expected_members = {
        path for path in available if path.name in assigned
    }
    if manifest_members != expected_members:
        missing = sorted(str(path) for path in expected_members - manifest_members)
        extra = sorted(str(path) for path in manifest_members - expected_members)
        raise SnapshotError(
            f"manifest whitelist mismatch; missing={missing}, extra={extra}"
        )
    selected: dict[str, bytes] = {}
    for record in records:
        data = available.get(record.archive_member)
        if data is None:
            raise SnapshotError(f"missing archive member: {record.archive_member}")
        if len(data) != record.size_bytes:
            raise SnapshotError(f"content size mismatch: {record.archive_member}")
        if hashlib.sha256(data).hexdigest() != record.content_sha256:
            raise SnapshotError(f"content hash mismatch: {record.archive_member}")
        expected_topic = assign_topics((record.archive_member.name,)).get(
            record.archive_member.name
        )
        if expected_topic != record.topic:
            raise SnapshotError(f"topic mismatch: {record.archive_member}")
        selected[record.archive_member.name] = data
    return selected


def _verify_existing_version(
    destination: Path, expected: dict[str, bytes]
) -> int:
    if not destination.is_dir() or destination.is_symlink():
        raise SnapshotConflictError(f"conflicting version path: {destination}")
    try:
        children = list(destination.iterdir())
    except OSError as error:
        raise SnapshotConflictError(
            f"unable to inspect version tree: {destination}"
        ) from error
    actual_paths: set[str] = set()
    for path in children:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise SnapshotConflictError(f"unable to inspect existing path: {path}") from error
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        attributes = getattr(metadata, "st_file_attributes", 0)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or bool(attributes & reparse)
        ):
            raise SnapshotConflictError(f"unexpected non-regular path: {path}")
        actual_paths.add(path.name)
    if actual_paths != set(expected):
        missing = set(expected) - actual_paths
        extra = actual_paths - set(expected)
        raise SnapshotConflictError(
            f"version tree conflict: {destination}; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for name, data in expected.items():
        target = destination / name
        try:
            actual = target.read_bytes()
        except OSError as error:
            raise SnapshotConflictError(f"unable to read existing file: {target}") from error
        if actual != data:
            raise SnapshotConflictError(f"conflicting existing file: {target.name}")
    return len(expected)


_DirectoryIdentity = tuple[int, int]


def _validate_directory(
    path: Path, expected_identity: _DirectoryIdentity | None = None
) -> _DirectoryIdentity:
    try:
        return validate_directory(path, expected_identity)
    except FileSafetyError as error:
        raise SnapshotError(f"unable to inspect staging directory: {path}") from error


def _validate_regular_metadata(metadata: os.stat_result, path: Path) -> None:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & reparse)
        or metadata.st_nlink != 1
    ):
        raise SnapshotError(f"unsafe staging file: {path}")


def _write_stage_file(stage: Path, name: str, data: bytes) -> None:
    """Create one staging file exclusively without following any existing link."""
    try:
        write_stage_file(stage, name, data)
    except FileSafetyError as error:
        raise SnapshotError(str(error)) from error


def _verify_stage(
    stage: Path,
    stage_identity: _DirectoryIdentity,
    expected: dict[str, bytes],
) -> None:
    _validate_directory(stage, stage_identity)
    try:
        children = list(stage.iterdir())
    except OSError as error:
        raise SnapshotError(f"unable to inspect staging directory: {stage}") from error
    if {path.name for path in children} != set(expected):
        raise SnapshotError(f"staging file set mismatch: {stage}")
    for path in children:
        try:
            metadata = path.lstat()
            _validate_regular_metadata(metadata, path)
        except OSError as error:
            raise SnapshotError(f"unable to inspect staging file: {path}") from error
        size, digest = _sha256_path(path)
        data = expected[path.name]
        if size != len(data) or digest != hashlib.sha256(data).hexdigest():
            raise SnapshotError(f"staging file verification failed: {path}")
    _validate_directory(stage, stage_identity)


def _prepare_stage(
    output_root: Path,
    output_root_identity: _DirectoryIdentity,
    version: str,
    expected: dict[str, bytes],
) -> tuple[Path, _DirectoryIdentity]:
    _validate_directory(output_root, output_root_identity)
    try:
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".postgresql-{version}.stage-", dir=output_root
            )
        )
    except OSError as error:
        raise SnapshotError("unable to create staging directory") from error
    _validate_directory(output_root, output_root_identity)
    identity = _validate_directory(stage)
    try:
        for name, data in sorted(expected.items()):
            _write_stage_file(stage, name, data)
        _verify_stage(stage, identity, expected)
    except Exception as error:
        _on_staging_failure(stage)
        raise SnapshotError(
            f"staging failed; path retained for manual inspection: {stage}: {error}"
        ) from error
    return stage, identity


def _on_staging_failure(_stage: Path) -> None:
    """Test hook; failed staging paths are deliberately left untouched."""


def _raise_retained_stage_error(error: Exception, stage: Path) -> None:
    message = f"{error}; staging retained for manual inspection: {stage}"
    if isinstance(error, SnapshotConflictError):
        raise SnapshotConflictError(message) from error
    raise SnapshotError(message) from error


@contextmanager
def _version_lock(
    output_root: Path, output_root_identity: _DirectoryIdentity, version: str
) -> Iterator[None]:
    try:
        _validate_directory(output_root, output_root_identity)
        with exclusive_lock(output_root, f"postgresql-{version}.restore"):
            yield
            _validate_directory(output_root, output_root_identity)
    except FileSafetyError as error:
        raise SnapshotError(f"unable to acquire restore lock: {output_root}") from error


def _ensure_output_root(output_root: Path) -> _DirectoryIdentity:
    try:
        return ensure_directory(output_root)
    except FileSafetyError as error:
        raise SnapshotError(f"unsafe output root: {output_root}") from error


def _same_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (metadata.st_dev, metadata.st_ino) == identity


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing any existing destination."""
    try:
        atomic_rename_directory_noreplace(source, destination)
    except FileConflictError as error:
        raise SnapshotConflictError(
            f"destination appeared during atomic publication: {destination}"
        ) from error
    except FileSafetyError as error:
        raise SnapshotError(str(error)) from error


def _before_atomic_publish(_stage: Path, _destination: Path) -> None:
    """Test hook immediately before the single atomic visibility transition."""


def _publish_atomic_version(
    output_root: Path,
    output_root_identity: _DirectoryIdentity,
    stage: Path,
    stage_identity: _DirectoryIdentity,
    destination: Path,
    expected: dict[str, bytes],
) -> _DirectoryIdentity:
    _validate_directory(output_root, output_root_identity)
    _verify_stage(stage, stage_identity, expected)
    if os.path.lexists(destination):
        raise SnapshotConflictError(f"destination already exists: {destination}")
    _before_atomic_publish(stage, destination)
    _validate_directory(output_root, output_root_identity)
    _validate_directory(stage, stage_identity)
    _atomic_rename_directory_noreplace(stage, destination)
    return stage_identity


def restore_snapshot(
    sources: Sequence[ArchiveSource],
    records: Sequence[SnapshotRecord],
    output_root: Path = Path("data/raw"),
    *,
    limits: ExtractionLimits = ExtractionLimits(),  # noqa: B008
) -> RestoreResult:
    """Restore each version atomically without overwriting conflicts.

    The transaction boundary is one PostgreSQL version, not the whole call. If
    a later version fails, an earlier version that already became completely
    visible remains in place and is safe to verify or skip on a subsequent run.
    No published path is automatically moved or deleted after publication.
    A failure before the atomic rename conservatively retains the randomly
    named staging directory; the exception reports its path for manual review.
    """
    source_by_version = _validate_record_sources(sources, records)
    grouped: dict[str, list[SnapshotRecord]] = {}
    for record in records:
        grouped.setdefault(record.postgresql_version, []).append(record)

    prepared: dict[str, dict[str, bytes]] = {}
    for version, version_records in sorted(grouped.items()):
        prepared[version] = _selected_bytes(
            source_by_version[version], version_records, limits
        )

    output_root_identity = _ensure_output_root(output_root)
    restored = 0
    skipped = 0
    absent: list[str] = []
    with ExitStack() as locks:
        for version in sorted(prepared):
            _validate_directory(output_root, output_root_identity)
            locks.enter_context(
                _version_lock(output_root, output_root_identity, version)
            )

        # Complete every conflict check possible before creating staging.
        for version, expected in sorted(prepared.items()):
            _validate_directory(output_root, output_root_identity)
            destination = output_root / f"postgresql-{version}"
            if os.path.lexists(destination):
                skipped += _verify_existing_version(destination, expected)
            else:
                absent.append(version)

        for version in absent:
            _validate_directory(output_root, output_root_identity)
            stage, identity = _prepare_stage(
                output_root, output_root_identity, version, prepared[version]
            )
            destination = output_root / f"postgresql-{version}"
            try:
                published_identity = _publish_atomic_version(
                    output_root,
                    output_root_identity,
                    stage,
                    identity,
                    destination,
                    prepared[version],
                )
            except (OSError, SnapshotError) as error:
                _on_staging_failure(stage)
                _raise_retained_stage_error(error, stage)
            _validate_directory(output_root, output_root_identity)
            if not _same_identity(destination, published_identity):
                raise SnapshotError(
                    f"published directory identity mismatch: {destination}"
                )
            _verify_existing_version(destination, prepared[version])
            restored += len(prepared[version])
    return RestoreResult(restored=restored, skipped=skipped)


def verify_snapshot(
    sources: Sequence[ArchiveSource],
    records: Sequence[SnapshotRecord],
    output_root: Path | None = None,
    *,
    limits: ExtractionLimits = ExtractionLimits(),  # noqa: B008
) -> None:
    """Verify archive bytes, record invariants, whitelist bytes, and raw output."""
    source_by_version = _validate_record_sources(sources, records)
    grouped: dict[str, list[SnapshotRecord]] = {}
    for record in records:
        grouped.setdefault(record.postgresql_version, []).append(record)
    for version, version_records in grouped.items():
        expected = _selected_bytes(source_by_version[version], version_records, limits)
        if output_root is not None:
            try:
                _verify_existing_version(
                    output_root / f"postgresql-{version}", expected
                )
            except SnapshotConflictError as error:
                raise SnapshotError(str(error)) from error


def _download_missing(sources: Iterable[ArchiveSource]) -> None:
    for source in sources:
        destination = Path(source.archive_path)
        if not destination.exists():
            download_archive(source, destination)


def _verify_sources_file(path: Path, sources: Sequence[ArchiveSource]) -> None:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotError(f"unable to read sources manifest: {path}") from error
    expected = [
        _source_dict(source)
        for source in sorted(sources, key=lambda item: item.version)
    ]
    if payload != expected:
        raise SnapshotError("sources manifest does not match pinned sources")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "restore"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--download-missing", action="store_true")
        if command == "restore":
            subparser.add_argument("--output-root", type=Path, default=Path("data/raw"))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-root", type=Path)
    return parser


def _run_command(args: argparse.Namespace) -> None:
    sources_path = Path("data/manifests/sources.json")
    snapshot_path = Path("data/manifests/snapshot.jsonl")
    if args.command == "freeze":
        if args.download_missing:
            _download_missing(SOURCES)
        records = tuple(
            record
            for source in SOURCES
            for record in build_snapshot_records(source, Path(source.archive_path))
        )
        write_sources_manifest(SOURCES, sources_path)
        write_snapshot_manifest(records, snapshot_path)
        return
    records = load_snapshot_manifest(snapshot_path)
    _verify_sources_file(sources_path, SOURCES)
    if args.command == "restore":
        if args.download_missing:
            _download_missing(SOURCES)
        restore_snapshot(SOURCES, records, args.output_root)
    else:
        verify_snapshot(SOURCES, records, args.output_root)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        _run_command(args)
    except (SnapshotError, DownloadError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
