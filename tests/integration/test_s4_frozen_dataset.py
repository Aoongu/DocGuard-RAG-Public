import json
from collections import Counter
from pathlib import Path

import pytest

from docguard.evaluation.dataset import (
    build_source_index,
    load_questions,
    sha256_file,
    validate_resolved_sources,
    validate_split_collection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "eval" / "s4-golden-v1"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "s2-chunk-v1" / "chunks.jsonl"


def test_s4_frozen_dataset_has_frozen_distribution_and_hashes() -> None:
    manifest = json.loads((DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    questions = []
    for split in ("development", "validation", "test"):
        path = DATASET_ROOT / f"{split}.jsonl"
        assert sha256_file(path) == manifest["files"][f"{split}.jsonl"]["sha256"]
        rows = load_questions(path)
        assert len(rows) == 60
        assert Counter(item.answerable for item in rows) == Counter({True: 30, False: 30})
        questions.extend(rows)

    validate_split_collection(questions)
    assert len(questions) == 180
    assert Counter(item.category for item in questions) == Counter(
        {
            "single_fact": 25,
            "how_to_parameter": 25,
            "cross_section": 20,
            "version_difference": 20,
            "out_of_domain": 30,
            "no_evidence_false_premise": 20,
            "unavailable_version": 20,
            "version_conflict_ambiguity": 20,
        }
    )
    assert Counter(item.source for item in questions) == Counter(
        {"assistant_curated": 144, "human_written": 36}
    )
    assert Counter(item.review_status for item in questions) == Counter(
        {"human_reviewed": 120, "double_reviewed": 60}
    )
    assert all(item.review_status == "double_reviewed" for item in questions if item.split == "test")
    assert manifest["status"] == "frozen"
    assert manifest["human_written_count"] == 36
    assert manifest["required_human_written_count"] == 36
    assert manifest["human_review_complete"] is True
    assert manifest["test_double_review_complete"] is True
    assert manifest["test_execution_status"] == "locked_until_s12"
    assert manifest["review_inputs"]["human_authored.jsonl"]["count"] == 36
    assert manifest["review_inputs"]["reviews_round1.jsonl"]["count"] == 180
    assert manifest["review_inputs"]["reviews_test_round2.jsonl"]["count"] == 60
    assert len(load_questions(PROJECT_ROOT / "data" / "eval" / "s4-evaluator-smoke-v1.jsonl")) == 30


@pytest.mark.skipif(not CHUNKS_PATH.exists(), reason="local S2 build artifact is unavailable")
def test_s4_frozen_source_refs_match_local_s2_chunks() -> None:
    questions = []
    for split in ("development", "validation", "test"):
        questions.extend(load_questions(DATASET_ROOT / f"{split}.jsonl"))
    validate_resolved_sources(questions, build_source_index(CHUNKS_PATH))
