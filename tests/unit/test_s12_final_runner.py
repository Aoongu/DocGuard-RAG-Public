from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from docguard.evaluation.dataset import load_questions
from docguard.evaluation.final_protocol import authorize_test_run, load_final_protocol
from docguard.evaluation.final_runner import run_final_evaluation
from docguard.generation.models import (
    GeneratedAnswer,
    GenerationTiming,
    GenerationUsage,
)
from docguard.refusal.models import Action, AnswerDecision, RefusalFeatures
from docguard.refusal.pipeline import M4Result

ROOT = Path(__file__).parents[2]
PROTOCOL = load_final_protocol(ROOT / "configs/evaluation/s12-final-v1.json")


def _features() -> RefusalFeatures:
    return RefusalFeatures(
        top1_rerank_score=2.0,
        top1_top2_margin=1.0,
        top3_mean_rerank_score=1.0,
        evidence_term_coverage=1.0,
        requested_version="18.6",
        requested_version_status="available",
        top_evidence_versions=("18.6",),
        best_score_by_version=(("18.6", 2.0),),
        paired_version_conflict=False,
        postgresql_domain_signal=True,
    )


class AnsweringPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, bool]] = []

    async def answer(
        self, question: str, requested_version: str | None, **switches: bool
    ) -> M4Result:
        del question, requested_version
        self.calls.append(dict(switches))
        decision = AnswerDecision(
            Action.ANSWER, None, 0.9, "thresholds_passed", _features()
        )
        generated = GeneratedAnswer(
            status="success",
            answer="A supported answer.",
            citations=(),
            evidence=(),
            validation_status="structural_only",
            raw_content='{"answer":"A supported answer.","citations":[]}',
            request_id="req-test",
            model="deepseek-flash",
            provider_fingerprint="fp-test",
            usage=GenerationUsage(100, 10, 20, 0.001, 0.0005, 0.001),
            timing_ms=GenerationTiming(1.0, 2.0, 1.0, 5.0, 1.0),
        )
        return M4Result(
            "m4", decision, generated, (), switches=tuple(sorted(switches.items()))
        )


def _setup(tmp_path: Path, method: str):
    test_path = tmp_path / PROTOCOL.dataset.test_path
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes((ROOT / PROTOCOL.dataset.test_path).read_bytes())
    authorization = authorize_test_run(tmp_path, PROTOCOL, method)
    frozen = PROTOCOL.methods[method]
    metadata = {
        "schema_version": 1,
        "protocol_version": PROTOCOL.protocol_version,
        "run_id": frozen.run_id,
        "split": "test",
        "dataset_version": PROTOCOL.dataset.version,
        "dataset_sha256": PROTOCOL.dataset.test_sha256,
        "method": method,
        "parameters": dict(frozen.parameters),
    }
    return authorization, metadata


def test_final_runner_records_switches_metrics_cost_and_never_repeats(
    tmp_path: Path,
) -> None:
    authorization, metadata = _setup(tmp_path, "m4_no_version_filter")
    questions = load_questions(ROOT / PROTOCOL.dataset.test_path)
    pipeline = AnsweringPipeline()

    first = asyncio.run(
        run_final_evaluation(
            questions,
            authorization.output_dir,
            pipeline,
            metadata,
            test_authorization=authorization,
        )
    )
    second = asyncio.run(
        run_final_evaluation(
            questions,
            authorization.output_dir,
            pipeline,
            metadata,
            test_authorization=authorization,
        )
    )

    assert first == second
    assert len(pipeline.calls) == 60
    assert set(pipeline.calls[0]) == {
        "refusal_enabled",
        "version_filter_enabled",
        "generation_enabled",
        "citation_validation_enabled",
    }
    assert pipeline.calls[0]["version_filter_enabled"] is False
    assert first["refusal"]["false_accept_count"] == 30
    assert first["generation"]["answerable"]["evaluated_count"] == 30
    assert first["operations"]["usage"]["api_cost_usd"] == pytest.approx(0.06)
    rows = [
        json.loads(line)
        for line in (authorization.output_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 60
    assert rows[0]["provider_fingerprint"] == "fp-test"
    assert rows[0]["parameters"]["version_filter_enabled"] is False
