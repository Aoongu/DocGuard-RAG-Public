from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from docguard.evaluation.refusal_cli import (
    load_examples,
    main,
    run_calibration,
    run_diagnostic,
    summarize_output,
)
from docguard.evaluation.runner import TestSplitLockedError
from docguard.refusal.config import load_refusal_config

ROOT = Path(__file__).resolve().parents[2]
FROZEN = load_refusal_config(ROOT / "configs" / "refusal" / "s8-refusal-v1.json")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    data = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, object]:
    dataset = tmp_path / "validation.jsonl"
    results = tmp_path / "results.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.json"
    dataset_hash = _write_jsonl(
        dataset,
        [
            {
                "question_id": "q1",
                "question": "According to PostgreSQL 17.11, what does SELECT do?",
                "split": "validation",
                "answerable": True,
                "expected_reason_code": None,
                "requested_version": "17.11",
                "category": "single_fact",
            }
        ],
    )
    results_hash = _write_jsonl(
        results,
        [
            {
                "question_id": "q1",
                "split": "validation",
                "status": "success",
                "hits": [
                    {"rank": 1, "chunk_id": "c1", "rerank_score": 2.0},
                    {"rank": 2, "chunk_id": "c2", "rerank_score": 1.0},
                    {"rank": 3, "chunk_id": "c3", "rerank_score": 0.0},
                ],
            }
        ],
    )
    chunks_hash = _write_jsonl(
        chunks,
        [
            {
                "chunk_id": f"c{index}",
                "postgresql_version": "17.11",
                "canonical_url": "https://www.postgresql.org/docs/17/sql-select.html",
                "anchor": f"A{index}",
                "text": "SELECT reads rows from a PostgreSQL table.",
            }
            for index in range(1, 4)
        ],
    )
    manifest.write_text(
        json.dumps({"chunks_sha256": chunks_hash}), encoding="utf-8"
    )
    identities = dict(FROZEN.input_sha256)
    identities["validation_dataset"] = dataset_hash
    identities["validation_m2_results"] = results_hash
    identities["s2_chunk_manifest"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config = replace(FROZEN, input_sha256=tuple(sorted(identities.items())))
    return dataset, results, chunks, manifest, config


def test_loader_joins_saved_hits_without_using_labels_as_features(tmp_path: Path) -> None:
    dataset, results, chunks, manifest, config = _fixture(tmp_path)
    examples = load_examples(dataset, results, chunks, manifest, "validation", config)
    assert len(examples) == 1
    assert examples[0].question_id == "q1"
    assert examples[0].features.top1_rerank_score == 2.0
    assert examples[0].features.requested_version == "17.11"


def test_calibration_writes_complete_immutable_report(tmp_path: Path) -> None:
    dataset, results, chunks, manifest, config = _fixture(tmp_path)
    examples = load_examples(dataset, results, chunks, manifest, "validation", config)
    output = tmp_path / "output"
    summary = run_calibration(examples, output, config, run_id="unit-selection")
    assert summary["result_status"] == "selection_result"
    assert summary["question_count"] == 1
    assert len((output / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert len(selection["candidates"]) == 1680
    with pytest.raises(ValueError):
        run_calibration(examples, output, config, run_id="changed")


def test_cli_rejects_test_before_reading_inputs(tmp_path: Path) -> None:
    with pytest.raises(TestSplitLockedError):
        main(
            [
                "calibrate",
                "--split",
                "test",
                "--dataset",
                str(tmp_path / "missing.jsonl"),
                "--results",
                str(tmp_path / "missing-results.jsonl"),
                "--chunks",
                str(tmp_path / "missing-chunks.jsonl"),
                "--chunk-manifest",
                str(tmp_path / "missing-manifest.json"),
                "--output",
                str(tmp_path / "output"),
            ]
        )


def test_diagnostic_reuses_selection_without_new_grid_search(tmp_path: Path) -> None:
    dataset, results, chunks, manifest, config = _fixture(tmp_path)
    examples = load_examples(dataset, results, chunks, manifest, "validation", config)
    selection_dir = tmp_path / "selection"
    run_calibration(examples, selection_dir, config, run_id="unit-selection")
    diagnostic_dir = tmp_path / "diagnostic"
    summary = run_diagnostic(
        examples,
        diagnostic_dir,
        config,
        selection_dir / "selection.json",
        run_id="unit-diagnostic",
        split="development",
    )
    assert summary["result_status"] == "diagnostic_replay"
    assert summary["selected_thresholds"] == json.loads(
        (selection_dir / "selection.json").read_text(encoding="utf-8")
    )["selected"]["thresholds"]
    checked = summarize_output(diagnostic_dir)
    assert checked["question_count"] == 1
    assert checked["recorded_count"] == 1
