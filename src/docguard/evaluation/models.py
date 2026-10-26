"""Strict immutable records used by the S4 evaluation framework."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

_VERSIONS = frozenset({"17.11", "18.6"})
_SPLITS = frozenset({"development", "validation", "test"})
_CATEGORIES = frozenset(
    {
        "single_fact",
        "how_to_parameter",
        "cross_section",
        "version_difference",
        "out_of_domain",
        "no_evidence_false_premise",
        "unavailable_version",
        "version_conflict_ambiguity",
    }
)
_REASON_CODES = frozenset(
    {"OUT_OF_DOMAIN", "NO_EVIDENCE", "UNSUPPORTED_VERSION", "VERSION_REQUIRED"}
)
_SOURCES = frozenset({"assistant_curated", "human_written"})
_REVIEW_STATUSES = frozenset(
    {"draft", "source_verified", "human_reviewed", "double_reviewed"}
)
_QUESTION_FIELDS = frozenset(
    {
        "schema_version",
        "question_id",
        "question",
        "requested_version",
        "answerable",
        "answer_key_points",
        "relevant_source_refs",
        "resolved_chunk_ids",
        "chunking_version",
        "expected_reason_code",
        "category",
        "split",
        "source",
        "review_status",
        "paraphrase_group_id",
    }
)
_SOURCE_FIELDS = frozenset({"version", "normalized_url", "anchor"})
_ID_RE = re.compile(r"s4-(?:dev|val|test)-\d{3}")
_CHUNK_RE = re.compile(r"chk_[0-9a-f]{64}")
_ANCHOR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")


def _strict_fields(data: dict[str, Any], expected: frozenset[str], label: str) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _non_empty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if "\r" in value or "\0" in value:
        raise ValueError(f"{field} must use LF text without NUL")
    return value


@dataclass(frozen=True, slots=True)
class SourceReference:
    version: str
    normalized_url: str
    anchor: str

    def __post_init__(self) -> None:
        if self.version not in _VERSIONS:
            raise ValueError("version must be 17.11 or 18.6")
        parsed = urlsplit(self.normalized_url)
        major = self.version.split(".", 1)[0]
        if (
            parsed.scheme != "https"
            or parsed.netloc != "www.postgresql.org"
            or not parsed.path.startswith(f"/docs/{major}/")
            or not parsed.path.endswith(".html")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("normalized_url must be an official versioned PostgreSQL URL")
        if not isinstance(self.anchor, str) or _ANCHOR_RE.fullmatch(self.anchor) is None:
            raise ValueError("anchor must be a safe non-empty HTML identifier")

    @classmethod
    def from_dict(cls, value: object) -> SourceReference:
        if not isinstance(value, dict):
            raise TypeError("source reference must be an object")
        _strict_fields(value, _SOURCE_FIELDS, "source reference")
        return cls(
            version=_non_empty(value["version"], "version"),
            normalized_url=_non_empty(value["normalized_url"], "normalized_url"),
            anchor=_non_empty(value["anchor"], "anchor"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "normalized_url": self.normalized_url,
            "anchor": self.anchor,
        }


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    schema_version: int
    question_id: str
    question: str
    requested_version: str | None
    answerable: bool
    answer_key_points: tuple[str, ...]
    relevant_source_refs: tuple[SourceReference, ...]
    resolved_chunk_ids: tuple[str, ...]
    chunking_version: str
    expected_reason_code: str | None
    category: str
    split: str
    source: str
    review_status: str
    paraphrase_group_id: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.question_id, str) or _ID_RE.fullmatch(self.question_id) is None:
            raise ValueError("question_id must match the frozen S4 ID format")
        _non_empty(self.question, "question")
        if self.requested_version is not None and self.requested_version not in _VERSIONS:
            raise ValueError("requested_version must be null, 17.11, or 18.6")
        if type(self.answerable) is not bool:
            raise ValueError("answerable must be a boolean")
        if self.category not in _CATEGORIES:
            raise ValueError("category is not supported")
        if self.split not in _SPLITS:
            raise ValueError("split is not supported")
        prefix = {"development": "s4-dev-", "validation": "s4-val-", "test": "s4-test-"}[self.split]
        if not self.question_id.startswith(prefix):
            raise ValueError("question_id prefix must match split")
        if self.chunking_version != "s2-chunk-v1":
            raise ValueError("chunking_version must be s2-chunk-v1")
        if self.source not in _SOURCES:
            raise ValueError("source is not supported")
        if self.review_status not in _REVIEW_STATUSES:
            raise ValueError("review_status is not supported")
        _non_empty(self.paraphrase_group_id, "paraphrase_group_id")
        if len(set(self.resolved_chunk_ids)) != len(self.resolved_chunk_ids):
            raise ValueError("resolved_chunk_ids must be unique")
        if any(_CHUNK_RE.fullmatch(item) is None for item in self.resolved_chunk_ids):
            raise ValueError("resolved_chunk_ids must contain full chunk IDs")
        if self.answerable:
            if self.expected_reason_code is not None:
                raise ValueError("answerable questions must not have a reason code")
            if not self.answer_key_points or not self.relevant_source_refs or not self.resolved_chunk_ids:
                raise ValueError("answerable questions require key points and evidence")
        else:
            if self.expected_reason_code not in _REASON_CODES:
                raise ValueError("boundary questions require a supported reason code")
            if self.answer_key_points or self.relevant_source_refs or self.resolved_chunk_ids:
                raise ValueError("boundary questions must not contain evidence")

    @classmethod
    def from_dict(cls, value: object) -> EvaluationQuestion:
        if not isinstance(value, dict):
            raise TypeError("evaluation question must be an object")
        _strict_fields(value, _QUESTION_FIELDS, "evaluation question")
        key_points = value["answer_key_points"]
        refs = value["relevant_source_refs"]
        chunk_ids = value["resolved_chunk_ids"]
        if not isinstance(key_points, list) or not all(isinstance(x, str) and x for x in key_points):
            raise ValueError("answer_key_points must be a list of non-empty strings")
        if not isinstance(refs, list):
            raise TypeError("relevant_source_refs must be a list")
        if not isinstance(chunk_ids, list) or not all(isinstance(x, str) for x in chunk_ids):
            raise ValueError("resolved_chunk_ids must be a list of strings")
        return cls(
            schema_version=value["schema_version"],
            question_id=_non_empty(value["question_id"], "question_id"),
            question=_non_empty(value["question"], "question"),
            requested_version=value["requested_version"],
            answerable=value["answerable"],
            answer_key_points=tuple(key_points),
            relevant_source_refs=tuple(SourceReference.from_dict(item) for item in refs),
            resolved_chunk_ids=tuple(chunk_ids),
            chunking_version=_non_empty(value["chunking_version"], "chunking_version"),
            expected_reason_code=value["expected_reason_code"],
            category=_non_empty(value["category"], "category"),
            split=_non_empty(value["split"], "split"),
            source=_non_empty(value["source"], "source"),
            review_status=_non_empty(value["review_status"], "review_status"),
            paraphrase_group_id=_non_empty(value["paraphrase_group_id"], "paraphrase_group_id"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "question_id": self.question_id,
            "question": self.question,
            "requested_version": self.requested_version,
            "answerable": self.answerable,
            "answer_key_points": list(self.answer_key_points),
            "relevant_source_refs": [item.to_dict() for item in self.relevant_source_refs],
            "resolved_chunk_ids": list(self.resolved_chunk_ids),
            "chunking_version": self.chunking_version,
            "expected_reason_code": self.expected_reason_code,
            "category": self.category,
            "split": self.split,
            "source": self.source,
            "review_status": self.review_status,
            "paraphrase_group_id": self.paraphrase_group_id,
        }

