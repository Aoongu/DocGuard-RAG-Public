import json

import pytest


def test_fixture_suite_is_frozen_disjoint_and_explicitly_not_human_reviewed(tmp_path):
    from docguard.discovery.evaluation_fixtures import load_suite, prepare_suite

    root = tmp_path / "suite"
    manifest = prepare_suite(root)
    assert manifest["kind"] == "synthetic_engineering_fixtures"
    assert manifest["human_expectations_status"] == "pending"
    assert manifest["test_execution_status"] == "locked"
    development = load_suite(root, "development")
    validation = load_suite(root, "validation")
    assert len(development) == len(validation) == 10
    assert {r["product"] for r in development}.isdisjoint(
        {r["product"] for r in validation}
    )
    assert {r["template_family"] for r in development}.isdisjoint(
        {r["template_family"] for r in validation}
    )
    with pytest.raises(ValueError, match="locked"):
        load_suite(root, "test")
    with pytest.raises(FileExistsError):
        prepare_suite(root)
    with (root / "development.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        load_suite(root, "development")


def test_summary_uses_fixed_denominators_and_unknown_usage_not_zero():
    from docguard.discovery.evaluation_metrics import summarize

    common = {
        "method": "adaptive_replay",
        "split": "development",
        "status": "failed",
        "expected_outcome": "discover",
        "outcome": "no_result",
        "qualified_count": 0,
        "candidate_count": 0,
        "source_correct_count": 0,
        "version_correct_count": 0,
        "imported_count": 0,
        "wrong_import_count": 0,
        "import_retrieval_success": False,
        "decision_steps": 2,
        "searches": 1,
        "reads": 0,
        "fetched_pages": 0,
        "latency_ms": 10.0,
        "llm_api_calls": 0,
        "model_cost_usd": None,
        "input_tokens": None,
        "output_tokens": None,
        "failure_reason_codes": ["tool_failure"],
        "model_usage_status": "unknown",
    }
    first = {
        **common,
        "case_id": "a",
        "status": "awaiting_selection",
        "outcome": "discover",
        "qualified_count": 1,
        "candidate_count": 1,
        "source_correct_count": 1,
        "version_correct_count": 1,
        "imported_count": 1,
        "import_retrieval_success": True,
        "failure_reason_codes": [],
    }
    report = summarize([first, {**common, "case_id": "b"}])[
        "development/adaptive_replay"
    ]
    assert report["discovery_success"]["denominator"] == 2
    assert report["discovery_success"]["numerator"] == 1
    assert report["import_retrieval_success"]["rate"] == 0.5
    assert report["source_accuracy"]["denominator"] == 1
    assert report["generation_quality_evaluation"] == "not_performed"
    assert report["model_cost_usd"] is None
    assert report["unknown_usage_rows"] == 2
    assert report["latency_ms"]["p95"] == 10.0
    assert report["failure_reason_counts"] == {"tool_failure": 1}
    assert report["fixed_sample_count"] == 2
    assert 0 < report["discovery_success"]["wilson_95_ci"][0] < 0.5
    excluded = summarize([{**common, "case_id": "c", "status": "not_applicable"}])[
        "development/adaptive_replay"
    ]
    assert excluded["fixed_sample_count"] == excluded["not_applicable_count"] == 1
    assert excluded["task_count"] == 0
    assert excluded["discovery_success"]["rate"] is None
    assert excluded["latency_ms"]["p95"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize([first, first])


@pytest.mark.asyncio
async def test_replay_uses_real_graph_approval_and_index_without_claiming_llm_quality(
    tmp_path,
):
    from pathlib import Path

    from docguard.discovery.evaluation import run_suite
    from docguard.discovery.evaluation_fixtures import prepare_suite

    root = tmp_path / "suite"
    prepare_suite(root)
    output = tmp_path / "run"
    report = await run_suite(
        root, output, ["development"], Path(__file__).resolve().parents[2]
    )
    assert report["evidence_level"] == "scripted_engineering_replay_only"
    assert report["human_validation_status"] == "pending"
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 30
    fixed = next(
        r
        for r in rows
        if r["scenario"] == "empty_then_rewrite"
        and r["method"] == "single_search_replay"
    )
    agent = next(
        r
        for r in rows
        if r["scenario"] == "empty_then_rewrite" and r["method"] == "adaptive_replay"
    )
    assert fixed["qualified_count"] == 0
    assert agent["import_retrieval_success"]
    assert agent["searches"] == 2
    assert agent["model_cost_usd"] is None
    assert agent["model_usage_status"] == "not_applicable_no_llm"
    assert report["metrics"]["development/adaptive_replay"]["unknown_usage_rows"] == 0
    assert agent["status"] == "ready"
    assert report["metrics"]["development/given_url"]["not_applicable_count"] == 4
    assert not agent["llm_api_calls"]
    assert (output / agent["trace_path"]).is_file()
    assert len(report["human_review_packet"]["reviews"]) == 20
    assert all(r["reviewer"] is None for r in report["human_review_packet"]["reviews"])
    with pytest.raises(FileExistsError):
        await run_suite(
            root, output, ["development"], Path(__file__).resolve().parents[2]
        )
    with pytest.raises(ValueError, match="locked"):
        await run_suite(
            root, tmp_path / "test-run", ["test"], Path(__file__).resolve().parents[2]
        )
