from __future__ import annotations

import json
from pathlib import Path

import pytest

from docguard.retrieval.hybrid_config import load_hybrid_config
from docguard.retrieval.technical_tokenizer import tokenize_technical

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "s5-hybrid-v1.json"


def test_committed_s5_config_freezes_hybrid_contract() -> None:
    config = load_hybrid_config(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.retrieval_version == "s5-hybrid-v1"
    assert config.tokenizer_version == "s5-tech-tokenizer-v1"
    assert config.bm25_enabled is True
    assert config.dense_enabled is True
    assert config.k1 == 1.2
    assert config.b == 0.75
    assert config.rrf_k_candidates == (20, 60)
    assert config.dense_depth_candidates == (20, 50)
    assert config.bm25_depth_candidates == (20, 50)
    assert config.selection_metrics == (
        "ndcg_at_10",
        "mrr_at_10",
        "recall_at_10",
        "recall_at_5",
    )
    assert config.chunking_version == "s2-chunk-v1"
    assert config.chunks_sha256 == (
        "6aad4667593d6519657cd1823b713983e09ed9f819e48246e0db9c18f0694964"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("extra", 1), "unknown fields"),
        (lambda value: value.pop("rrf"), "missing fields"),
        (lambda value: value.__setitem__("bm25_enabled", False), "must be true"),
        (lambda value: value["bm25"].__setitem__("k1", float("nan")), "finite"),
        (lambda value: value["rrf"].__setitem__("k_candidates", [60, 20]), "[20, 60]"),
        (lambda value: value.__setitem__("chunks_sha256", "bad"), "SHA-256"),
    ],
)
def test_s5_config_rejects_contract_drift(tmp_path: Path, mutation, message: str) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutation(value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_hybrid_config(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("wal_level", ("wal_level", "wal", "level")),
        ("pg_hba.conf", ("pg_hba.conf", "pg", "hba", "conf")),
        ("--max-connections", ("--max-connections", "max", "connections")),
        ("SQLSTATE 23505", ("sqlstate", "23505")),
        ("PostgreSQL 17.11", ("postgresql", "17.11", "17", "11")),
        ("CREATE INDEX", ("create", "index")),
        (
            "duplicate key value violates unique constraint",
            ("duplicate", "key", "value", "violates", "unique", "constraint"),
        ),
        ("ＷＡＬ＿ＬＥＶＥＬ", ("wal_level", "wal", "level")),
        ("  ", ()),
    ],
)
def test_technical_tokenizer_has_exact_deterministic_output(
    text: str, expected: tuple[str, ...]
) -> None:
    assert tokenize_technical(text) == expected
    assert tokenize_technical(text) == expected


def test_technical_tokenizer_preserves_repeated_occurrences() -> None:
    assert tokenize_technical("wal_level WAL_LEVEL") == (
        "wal_level",
        "wal",
        "level",
        "wal_level",
        "wal",
        "level",
    )


def test_technical_tokenizer_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        tokenize_technical(None)  # type: ignore[arg-type]
