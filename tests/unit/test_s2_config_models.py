import hashlib
import json
import unicodedata
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from docguard.processing.config import load_chunking_config
from docguard.processing.models import (
    ChunkRecord,
    DocumentRecord,
    SectionBlock,
    make_chunk_id,
    make_document_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "chunking" / "s2-chunk-v1.json"


def _document(**changes: object) -> DocumentRecord:
    values: dict[str, object] = {
        "document_id": make_document_id(
            "17.11", "https://www.postgresql.org/docs/17/sql-select.html"
        ),
        "postgresql_version": "17.11",
        "topic": "sql_commands",
        "canonical_url": "https://www.postgresql.org/docs/17/sql-select.html",
        "title": "SELECT",
        "content_sha256": hashlib.sha256(b"document").hexdigest(),
    }
    values.update(changes)
    return DocumentRecord(**values)  # type: ignore[arg-type]


def _chunk(**changes: object) -> ChunkRecord:
    text = "SELECT retrieves rows from a table."
    url = "https://www.postgresql.org/docs/17/sql-select.html"
    values: dict[str, object] = {
        "schema_version": 1,
        "chunk_id": make_chunk_id("s2-chunk-v1", "17.11", url, "SQL-SELECT", 0),
        "document_id": make_document_id("17.11", url),
        "postgresql_version": "17.11",
        "topic": "sql_commands",
        "canonical_url": url,
        "anchor": "SQL-SELECT",
        "section_path": ("SQL Commands", "SELECT"),
        "ordinal": 0,
        "text": text,
        "token_count": 9,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chunking_version": "s2-chunk-v1",
    }
    values.update(changes)
    return ChunkRecord(**values)  # type: ignore[arg-type]


def test_frozen_config_pins_models_and_token_budget() -> None:
    config = load_chunking_config(CONFIG_PATH)

    assert config.chunking_version == "s2-chunk-v1"
    assert config.max_chunk_tokens == 384
    assert config.overlap_tokens == 48
    assert config.min_chunk_tokens == 64
    assert config.query_budget == 96
    assert config.special_tokens_budget == 4
    assert config.safety_margin == 28
    assert config.html_normalization_version == "postgresql-html-v1"
    assert config.id_rule_version == "sha256-nul-v1"
    assert config.embedding.name == "intfloat/multilingual-e5-small"
    assert config.embedding.revision == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert config.embedding.max_length == 512
    assert config.reranker.name == "BAAI/bge-reranker-base"
    assert config.reranker.revision == "2cfc18c9415c912f9d8155881c133215df768a70"
    assert config.reranker.max_length == 512
    with pytest.raises(FrozenInstanceError):
        config.max_chunk_tokens = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update({"unknown": True}), "unknown"),
        (lambda data: data.pop("chunking_version"), "chunking_version"),
        (lambda data: data.update({"max_chunk_tokens": 385}), "budget"),
        (lambda data: data.update({"overlap_tokens": 384}), "overlap"),
        (
            lambda data: data["embedding"].update({"revision": "main"}),
            "revision",
        ),
        (
            lambda data: data["embedding"].update({"unknown": True}),
            "unknown",
        ),
    ],
)
def test_config_rejects_invalid_or_unknown_fields(
    tmp_path: Path, mutation: object, message: str
) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutation(data)  # type: ignore[operator]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_chunking_config(path)


def test_ids_use_nul_delimited_full_sha256_and_isolate_versions() -> None:
    url = "https://www.postgresql.org/docs/17/sql-select.html"
    expected_document = "doc_" + hashlib.sha256(f"17.11\0{url}".encode()).hexdigest()
    separator = "\0"
    expected_chunk = "chk_" + hashlib.sha256(
        separator.join(("s2-chunk-v1", "17.11", url, "SQL-SELECT", "0")).encode(
            "utf-8"
        )
    ).hexdigest()

    assert make_document_id("17.11", url) == expected_document
    assert make_chunk_id("s2-chunk-v1", "17.11", url, "SQL-SELECT", 0) == expected_chunk
    assert make_document_id("18.6", url) != expected_document
    assert make_chunk_id("s2-chunk-v2", "17.11", url, "SQL-SELECT", 0) != expected_chunk


def test_models_are_immutable_slotted_and_accept_canonical_values() -> None:
    block = SectionBlock("paragraph", "SQL-SELECT", ("SELECT",), "Canonical text")
    document = _document()
    chunk = _chunk()

    assert not hasattr(block, "__dict__")
    assert not hasattr(document, "__dict__")
    assert not hasattr(chunk, "__dict__")
    with pytest.raises(FrozenInstanceError):
        block.text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"document_id": "doc_" + "0" * 64}, "document_id"),
        ({"postgresql_version": "17"}, "postgresql_version"),
        ({"topic": "other"}, "topic"),
        ({"canonical_url": "http://example.com/a.html"}, "canonical_url"),
        ({"title": ""}, "title"),
        ({"title": unicodedata.normalize("NFD", "Café")}, "NFC"),
        ({"content_sha256": "A" * 64}, "content_sha256"),
    ],
)
def test_document_rejects_invalid_values(changes: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _document(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"chunk_id": "chk_" + "0" * 64}, "chunk_id"),
        ({"document_id": "doc_" + "0" * 64}, "document_id"),
        ({"anchor": ""}, "anchor"),
        ({"anchor": "bad anchor"}, "anchor"),
        ({"section_path": ()}, "section_path"),
        ({"ordinal": -1}, "ordinal"),
        ({"text": ""}, "text"),
        ({"text": unicodedata.normalize("NFD", "Café")}, "NFC"),
        ({"token_count": 0}, "token_count"),
        ({"content_sha256": "0" * 64}, "content_sha256"),
        ({"chunking_version": "other"}, "chunking_version"),
        ({"canonical_url": "https://example.com/a.html"}, "canonical_url"),
    ],
)
def test_chunk_rejects_invalid_or_inconsistent_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _chunk(**changes)


def test_replacing_text_requires_matching_content_hash() -> None:
    chunk = _chunk()

    with pytest.raises(ValueError, match="content_sha256"):
        replace(chunk, text="different canonical text")

