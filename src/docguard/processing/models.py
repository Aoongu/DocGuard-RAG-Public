"""Immutable normalized records and stable identifiers for S2."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from docguard.ingestion.models import POSTGRESQL_VERSIONS, SNAPSHOT_TOPICS

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DOCUMENT_ID_RE = re.compile(r"doc_[0-9a-f]{64}")
_CHUNK_ID_RE = re.compile(r"chk_[0-9a-f]{64}")
_ANCHOR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_CHUNKING_VERSION_RE = re.compile(r"s2-chunk-v[1-9]\d*")
_BLOCK_KINDS = frozenset({"paragraph", "list", "code", "table"})


def _digest_id(prefix: str, parts: tuple[str, ...]) -> str:
    if any(not isinstance(part, str) for part in parts):
        raise TypeError("ID inputs must be strings")
    return prefix + hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def make_document_id(postgresql_version: str, canonical_url: str) -> str:
    """Return the full digest ID for one frozen source document."""
    return _digest_id("doc_", (postgresql_version, canonical_url))


def make_chunk_id(
    chunking_version: str,
    postgresql_version: str,
    canonical_url: str,
    anchor: str,
    ordinal: int,
) -> str:
    """Return the full digest ID for one chunk position."""
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("ordinal must be a non-negative integer")
    return _digest_id(
        "chk_",
        (
            chunking_version,
            postgresql_version,
            canonical_url,
            anchor,
            str(ordinal),
        ),
    )


def _validate_nfc_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field_name} must use NFC normalization")
    if "\r" in value or "\0" in value:
        raise ValueError(f"{field_name} must use LF text without NUL")
    return value


def _validate_anchor(anchor: object) -> str:
    if not isinstance(anchor, str) or _ANCHOR_RE.fullmatch(anchor) is None:
        raise ValueError("anchor must be a non-empty safe HTML identifier")
    return anchor


def _validate_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _validate_source(
    postgresql_version: object, topic: object, canonical_url: object
) -> tuple[str, str, str]:
    if postgresql_version not in POSTGRESQL_VERSIONS:
        raise ValueError("postgresql_version must be a pinned version")
    if topic not in SNAPSHOT_TOPICS:
        raise ValueError("topic must be a supported technical topic")
    if not isinstance(canonical_url, str):
        raise TypeError("canonical_url must be a string")
    parsed = urlsplit(canonical_url)
    major = str(postgresql_version).split(".", 1)[0]
    expected_prefix = f"/docs/{major}/"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "www.postgresql.org"
        or not parsed.path.startswith(expected_prefix)
        or not parsed.path.endswith(".html")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("canonical_url must be an official versioned docs URL")
    return str(postgresql_version), str(topic), canonical_url


def _validate_section_path(section_path: object) -> tuple[str, ...]:
    if not isinstance(section_path, tuple) or not section_path:
        raise ValueError("section_path must be a non-empty tuple")
    for item in section_path:
        _validate_nfc_text(item, "section_path item")
    return section_path


@dataclass(frozen=True, slots=True)
class SectionBlock:
    kind: Literal["paragraph", "list", "code", "table"]
    anchor: str
    section_path: tuple[str, ...]
    text: str

    def __post_init__(self) -> None:
        if self.kind not in _BLOCK_KINDS:
            raise ValueError("kind must be paragraph, list, code, or table")
        _validate_anchor(self.anchor)
        _validate_section_path(self.section_path)
        _validate_nfc_text(self.text, "text")


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    postgresql_version: str
    topic: str
    canonical_url: str
    title: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or _DOCUMENT_ID_RE.fullmatch(
            self.document_id
        ) is None:
            raise ValueError("document_id must be a full prefixed SHA-256")
        _validate_source(self.postgresql_version, self.topic, self.canonical_url)
        expected = make_document_id(self.postgresql_version, self.canonical_url)
        if self.document_id != expected:
            raise ValueError("document_id does not match source identity")
        _validate_nfc_text(self.title, "title")
        _validate_sha256(self.content_sha256, "content_sha256")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    schema_version: int
    chunk_id: str
    document_id: str
    postgresql_version: str
    topic: str
    canonical_url: str
    anchor: str
    section_path: tuple[str, ...]
    ordinal: int
    text: str
    token_count: int
    content_sha256: str
    chunking_version: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.chunk_id, str) or _CHUNK_ID_RE.fullmatch(
            self.chunk_id
        ) is None:
            raise ValueError("chunk_id must be a full prefixed SHA-256")
        if not isinstance(self.document_id, str) or _DOCUMENT_ID_RE.fullmatch(
            self.document_id
        ) is None:
            raise ValueError("document_id must be a full prefixed SHA-256")
        _validate_source(self.postgresql_version, self.topic, self.canonical_url)
        if self.document_id != make_document_id(
            self.postgresql_version, self.canonical_url
        ):
            raise ValueError("document_id does not match source identity")
        _validate_anchor(self.anchor)
        _validate_section_path(self.section_path)
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if (
            not isinstance(self.chunking_version, str)
            or _CHUNKING_VERSION_RE.fullmatch(self.chunking_version) is None
        ):
            raise ValueError("chunking_version must be a supported S2 version")
        expected_chunk_id = make_chunk_id(
            self.chunking_version,
            self.postgresql_version,
            self.canonical_url,
            self.anchor,
            self.ordinal,
        )
        if self.chunk_id != expected_chunk_id:
            raise ValueError("chunk_id does not match chunk identity")
        text = _validate_nfc_text(self.text, "text")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("token_count must be a positive integer")
        _validate_sha256(self.content_sha256, "content_sha256")
        expected_content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self.content_sha256 != expected_content_sha256:
            raise ValueError("content_sha256 does not match text")
