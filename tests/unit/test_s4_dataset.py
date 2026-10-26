import hashlib
import json
from pathlib import Path

import pytest

from docguard.evaluation.dataset import (
    DatasetValidationError,
    build_source_index,
    load_questions,
    resolve_question_sources,
    sha256_file,
    validate_split_collection,
)


def _chunk(chunk_id: str, version: str, url: str, anchor: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "chunk_id": chunk_id,
        "postgresql_version": version,
        "canonical_url": url,
        "anchor": anchor,
    }


def _question(question_id: str, split: str, group: str) -> dict[str, object]:
    prefix = {"development": "dev", "validation": "val", "test": "test"}[split]
    question_id = f"s4-{prefix}-{question_id}"
    return {
        "schema_version": 1,
        "question_id": question_id,
        "question": "What does SELECT do?",
        "requested_version": "18.6",
        "answerable": True,
        "answer_key_points": ["It retrieves rows."],
        "relevant_source_refs": [
            {
                "version": "18.6",
                "normalized_url": "https://www.postgresql.org/docs/18/sql-select.html",
                "anchor": "SQL-SELECT",
            }
        ],
        "resolved_chunk_ids": ["chk_" + "a" * 64],
        "chunking_version": "s2-chunk-v1",
        "expected_reason_code": None,
        "category": "single_fact",
        "split": split,
        "source": "assistant_curated",
        "review_status": "source_verified",
        "paraphrase_group_id": group,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_load_questions_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    row = _question("001", "development", "select")
    _write_jsonl(path, [row, row])

    with pytest.raises(DatasetValidationError, match="duplicate question_id"):
        load_questions(path)


def test_collection_rejects_paraphrase_group_cross_split() -> None:
    dev = [_question("001", "development", "same-fact")]
    val = [_question("001", "validation", "same-fact")]

    with pytest.raises(DatasetValidationError, match="paraphrase_group_id"):
        validate_split_collection(
            [
                *[load for load in _as_questions(dev)],
                *[load for load in _as_questions(val)],
            ]
        )


def _as_questions(rows: list[dict[str, object]]):
    from docguard.evaluation.models import EvaluationQuestion

    return [EvaluationQuestion.from_dict(row) for row in rows]


def test_source_refs_resolve_deterministically_and_require_exact_match(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    url = "https://www.postgresql.org/docs/18/sql-select.html"
    first = "chk_" + "1" * 64
    second = "chk_" + "2" * 64
    _write_jsonl(
        chunks,
        [
            _chunk(second, "18.6", url, "SQL-SELECT"),
            _chunk(first, "18.6", url, "SQL-SELECT"),
        ],
    )
    index = build_source_index(chunks)
    question = _as_questions([_question("001", "development", "select")])[0]

    resolved = resolve_question_sources(question, index)

    assert resolved == (first, second)
    missing = _question("002", "development", "missing")
    missing["relevant_source_refs"][0]["anchor"] = "MISSING"  # type: ignore[index]
    with pytest.raises(DatasetValidationError, match="does not resolve"):
        resolve_question_sources(_as_questions([missing])[0], index)


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_bytes(b"one\ntwo\n")

    assert sha256_file(path) == hashlib.sha256(b"one\ntwo\n").hexdigest()
