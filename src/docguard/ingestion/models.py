"""Immutable data models for PostgreSQL documentation snapshots."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from urllib.parse import SplitResult, urlsplit

POSTGRESQL_VERSIONS = frozenset({"17.11", "18.6"})
SNAPSHOT_TOPICS = frozenset(
    {
        "auth_privileges",
        "backup_restore",
        "installation_configuration",
        "performance_tuning",
        "sql_commands",
        "sql_language",
    }
)

_VERSION_RE = re.compile(r"[1-9]\d*\.\d+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UTC_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_LOCAL_HTTP_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _validate_version(version: str, field_name: str = "version") -> None:
    if not isinstance(version, str) or _VERSION_RE.fullmatch(version) is None:
        raise ValueError(f"{field_name} must be a major.minor version")


def _split_url(url: str, field_name: str) -> SplitResult:
    if not isinstance(url, str):
        raise TypeError(f"{field_name} must be a string")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid URL") from error
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain credentials")
    if port is not None and not 0 < port <= 65_535:
        raise ValueError(f"{field_name} has an invalid port")
    return parsed


def _validate_https_url(url: str, field_name: str) -> None:
    parsed = _split_url(url, field_name)
    if parsed.scheme != "https":
        raise ValueError(f"{field_name} must be an HTTPS URL")


def _validate_archive_url(url: str) -> None:
    parsed = _split_url(url, "url")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in _LOCAL_HTTP_HOSTS:
        return
    raise ValueError("url must use HTTPS or local test-server HTTP")


def _validate_relative_posix_path(path: PurePosixPath, field_name: str) -> None:
    if not isinstance(path, PurePosixPath):
        raise TypeError(f"{field_name} must be a PurePosixPath")
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or "\\" in str(path)
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"{field_name} must be a repository-relative POSIX path")


def _validate_positive_size(size_bytes: int) -> None:
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ValueError("size_bytes must be a positive integer")


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ArchiveSource:
    version: str
    major_version: str
    url: str
    archive_path: PurePosixPath
    size_bytes: int
    sha256: str
    retrieved_at: str
    license_name: str = "PostgreSQL"
    license_url: str = "https://www.postgresql.org/about/licence/"

    def __post_init__(self) -> None:
        _validate_version(self.version)
        if (
            not isinstance(self.major_version, str)
            or self.major_version != self.version.split(".", 1)[0]
        ):
            raise ValueError("major_version must match version")
        _validate_archive_url(self.url)
        _validate_relative_posix_path(self.archive_path, "archive_path")
        _validate_positive_size(self.size_bytes)
        _validate_sha256(self.sha256, "sha256")
        if (
            not isinstance(self.retrieved_at, str)
            or _UTC_TIMESTAMP_RE.fullmatch(self.retrieved_at) is None
        ):
            raise ValueError("retrieved_at must be an ISO 8601 UTC Z timestamp")
        try:
            datetime.strptime(self.retrieved_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as error:
            raise ValueError("retrieved_at must be a valid UTC Z timestamp") from error
        if self.license_name != "PostgreSQL":
            raise ValueError("license_name must be PostgreSQL")
        _validate_https_url(self.license_url, "license_url")


@dataclass(frozen=True, slots=True)
class TopicRule:
    name: str
    patterns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if (
            not isinstance(self.patterns, tuple)
            or not self.patterns
            or any(
                not isinstance(pattern, str) or not pattern
                for pattern in self.patterns
            )
        ):
            raise ValueError("patterns must be a non-empty tuple of strings")


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    schema_version: int
    postgresql_version: str
    topic: str
    archive_path: PurePosixPath
    archive_sha256: str
    archive_member: PurePosixPath
    raw_path: PurePosixPath
    canonical_url: str
    content_sha256: str
    size_bytes: int
    media_type: str = "text/html"
    license: str = "PostgreSQL"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.postgresql_version not in POSTGRESQL_VERSIONS:
            raise ValueError("postgresql_version must be a pinned version")
        if self.topic not in SNAPSHOT_TOPICS:
            raise ValueError("topic must be a supported technical topic")
        _validate_relative_posix_path(self.archive_path, "archive_path")
        expected_archive_path = PurePosixPath(
            "data",
            "archives",
            f"postgresql-{self.postgresql_version}-docs.tar.gz",
        )
        if self.archive_path != expected_archive_path:
            raise ValueError("archive_path must match postgresql_version")
        _validate_sha256(self.archive_sha256, "archive_sha256")
        _validate_relative_posix_path(self.archive_member, "archive_member")
        member_root = PurePosixPath(
            f"postgresql-{self.postgresql_version}",
            "doc",
            "src",
            "sgml",
            "html",
        )
        if (
            not self.archive_member.is_relative_to(member_root)
            or self.archive_member == member_root
        ):
            raise ValueError("archive_member must use the versioned HTML prefix")
        _validate_relative_posix_path(self.raw_path, "raw_path")
        raw_root = PurePosixPath(
            "data", "raw", f"postgresql-{self.postgresql_version}"
        )
        if self.raw_path.parent != raw_root:
            raise ValueError("raw_path must be directly under the versioned raw directory")
        canonical = _split_url(self.canonical_url, "canonical_url")
        major_version = self.postgresql_version.split(".", 1)[0]
        if (
            canonical.scheme != "https"
            or canonical.netloc != "www.postgresql.org"
            or canonical.query
            or canonical.fragment
            or "?" in self.canonical_url
            or "#" in self.canonical_url
        ):
            raise ValueError("canonical_url must be an official versioned docs URL")
        basenames = {
            self.archive_member.name,
            self.raw_path.name,
            canonical.path.rsplit("/", 1)[-1],
        }
        if len(basenames) != 1:
            raise ValueError("archive member, raw path, and canonical URL basename differ")
        basename = basenames.pop()
        if not basename.endswith(".html"):
            raise ValueError("basename must end with lowercase .html")
        expected = f"https://www.postgresql.org/docs/{major_version}/{basename}"
        if self.canonical_url != expected:
            raise ValueError("canonical_url must exactly match the canonical docs URL")
        _validate_sha256(self.content_sha256, "content_sha256")
        _validate_positive_size(self.size_bytes)
        if self.media_type != "text/html":
            raise ValueError("media_type must be text/html")
        if self.license != "PostgreSQL":
            raise ValueError("license must be PostgreSQL")
