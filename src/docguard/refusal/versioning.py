"""Deterministic version request and cross-version conflict analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from docguard.refusal.config import RefusalConfig
from docguard.refusal.models import RankedEvidence

_POSTGRES_VERSION = re.compile(r"\bPostgreSQL\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_COMPARISON_WORD = re.compile(r"\b(compare|comparison|versus|vs\.?|both)\b", re.IGNORECASE)
_VERSION_NUMBER = re.compile(r"\b\d{1,2}(?:\.\d+)?\b")
_WORD = re.compile(r"[a-z_][a-z0-9_]{2,}")


@dataclass(frozen=True, slots=True)
class VersionRequest:
    requested_version: str | None
    status: str
    mentioned_versions: tuple[str, ...]
    comparison_requested: bool


@dataclass(frozen=True, slots=True)
class VersionConflict:
    detected: bool
    source_key: str | None
    token_jaccard: float | None
    versions: tuple[str, ...]


def parse_version_request(
    question: str,
    structured_version: str | None,
    config: RefusalConfig,
) -> VersionRequest:
    if not isinstance(question, str) or not question.strip() or "\0" in question:
        raise ValueError("question must be a non-empty string without NUL")
    if structured_version is not None:
        if not isinstance(structured_version, str) or not structured_version.strip():
            raise ValueError("structured_version must be null or a non-empty string")
        value = structured_version.strip()
        status = "available" if value in config.supported_versions else "unavailable"
        return VersionRequest(value, status, (value,), False)
    mentioned = tuple(dict.fromkeys(_POSTGRES_VERSION.findall(question)))
    has_comparison_word = bool(_COMPARISON_WORD.search(question))
    if mentioned and has_comparison_word:
        mentioned = tuple(dict.fromkeys(_VERSION_NUMBER.findall(question)))
    comparison = len(mentioned) > 1 and has_comparison_word
    if len(mentioned) == 1:
        value = mentioned[0]
        status = "available" if value in config.supported_versions else "unavailable"
        return VersionRequest(value, status, mentioned, False)
    if len(mentioned) > 1:
        return VersionRequest(None, "ambiguous", mentioned, comparison)
    return VersionRequest(None, "none", (), False)


def _source_key(hit: RankedEvidence) -> str:
    parts = urlsplit(hit.canonical_url)
    path = re.sub(r"/docs/(?:17|18)/", "/docs/{version}/", parts.path)
    base = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))
    return f"{base}#{hit.anchor}"


def _text_tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(left: str, right: str) -> float:
    left_tokens = _text_tokens(left)
    right_tokens = _text_tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def detect_version_conflict(
    hits: list[RankedEvidence], config: RefusalConfig
) -> VersionConflict:
    grouped: dict[str, dict[str, RankedEvidence]] = {}
    for hit in hits:
        if hit.postgresql_version not in config.supported_versions:
            raise ValueError("hit version is outside the frozen corpus")
        if hit.rerank_score < config.version_conflict.min_score:
            continue
        versions = grouped.setdefault(_source_key(hit), {})
        current = versions.get(hit.postgresql_version)
        if current is None or hit.rerank_score > current.rerank_score:
            versions[hit.postgresql_version] = hit
    for source_key in sorted(grouped):
        by_version = grouped[source_key]
        if set(by_version) != set(config.supported_versions):
            continue
        left, right = (by_version[version] for version in config.supported_versions)
        similarity = _jaccard(left.text, right.text)
        if similarity < config.version_conflict.max_token_jaccard:
            return VersionConflict(True, source_key, similarity, config.supported_versions)
    return VersionConflict(False, None, None, ())
