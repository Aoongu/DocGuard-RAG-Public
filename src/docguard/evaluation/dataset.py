"""Dataset loading, stable source resolution, and leakage validation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from docguard.evaluation.models import EvaluationQuestion, SourceReference
from docguard.fs_safety import validate_regular_file

SourceKey = tuple[str, str, str]


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset violates its frozen contract."""


def sha256_file(path: Path) -> str:
    validate_regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions(path: Path) -> list[EvaluationQuestion]:
    validate_regular_file(path)
    questions: list[EvaluationQuestion] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise DatasetValidationError(f"blank JSONL line at {line_number}")
        try:
            question = EvaluationQuestion.from_dict(json.loads(line))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise DatasetValidationError(f"invalid question at line {line_number}: {error}") from error
        if question.question_id in seen:
            raise DatasetValidationError(
                f"duplicate question_id at line {line_number}: {question.question_id}"
            )
        seen.add(question.question_id)
        questions.append(question)
    if not questions:
        raise DatasetValidationError("evaluation dataset must not be empty")
    return questions


def validate_split_collection(questions: list[EvaluationQuestion]) -> None:
    ids: set[str] = set()
    groups: dict[str, str] = {}
    for question in questions:
        if question.question_id in ids:
            raise DatasetValidationError(f"duplicate question_id: {question.question_id}")
        ids.add(question.question_id)
        existing_split = groups.setdefault(question.paraphrase_group_id, question.split)
        if existing_split != question.split:
            raise DatasetValidationError(
                "paraphrase_group_id crosses splits: "
                f"{question.paraphrase_group_id} ({existing_split}, {question.split})"
            )


def _source_key(reference: SourceReference) -> SourceKey:
    return (reference.version, reference.normalized_url, reference.anchor)


def build_source_index(chunks_path: Path) -> dict[SourceKey, tuple[str, ...]]:
    validate_regular_file(chunks_path)
    collected: defaultdict[SourceKey, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    for line_number, line in enumerate(chunks_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetValidationError(f"invalid chunk JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise DatasetValidationError(f"chunk line {line_number} must be an object")
        required = ("chunk_id", "postgresql_version", "canonical_url", "anchor")
        if any(not isinstance(row.get(field), str) or not row[field] for field in required):
            raise DatasetValidationError(f"chunk line {line_number} lacks source identity")
        chunk_id = row["chunk_id"]
        if chunk_id in seen_ids:
            raise DatasetValidationError(f"duplicate chunk_id at line {line_number}: {chunk_id}")
        seen_ids.add(chunk_id)
        key = (row["postgresql_version"], row["canonical_url"], row["anchor"])
        collected[key].append(chunk_id)
    if not collected:
        raise DatasetValidationError("chunk source index must not be empty")
    return {key: tuple(sorted(values)) for key, values in sorted(collected.items())}


def resolve_question_sources(
    question: EvaluationQuestion,
    source_index: dict[SourceKey, tuple[str, ...]],
) -> tuple[str, ...]:
    if not question.answerable:
        return ()
    chunk_ids: set[str] = set()
    for reference in question.relevant_source_refs:
        key = _source_key(reference)
        matches = source_index.get(key)
        if not matches:
            raise DatasetValidationError(
                "source reference does not resolve: " + " | ".join(key)
            )
        chunk_ids.update(matches)
    return tuple(sorted(chunk_ids))


def validate_resolved_sources(
    questions: list[EvaluationQuestion],
    source_index: dict[SourceKey, tuple[str, ...]],
) -> None:
    for question in questions:
        resolved = resolve_question_sources(question, source_index)
        if resolved != tuple(sorted(question.resolved_chunk_ids)):
            raise DatasetValidationError(
                f"resolved_chunk_ids mismatch for {question.question_id}"
            )
