import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from docguard.evaluation.dataset import load_questions
from docguard.evaluation.final_protocol import authorize_test_run, load_final_protocol
from docguard.evaluation.generation_runner import run_generation_evaluation
from docguard.evaluation.models import EvaluationQuestion
from docguard.evaluation.runner import RunContractError, TestSplitLockedError
from docguard.generation.client import FixedGenerationClient, GenerationUpstreamError
from docguard.generation.config import load_generation_config
from docguard.generation.models import (
    Evidence,
    EvidenceProvenance,
    GeneratedAnswer,
    GenerationCompletion,
    GenerationTiming,
    GenerationUsage,
    VerifiedCitation,
)
from docguard.generation.pipeline import B2Pipeline
from docguard.generation.service import GenerationService
from docguard.retrieval.service import DenseSearchHit

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/generation/s7-generation-v1.json"
S12_PROTOCOL = load_final_protocol(ROOT / "configs/evaluation/s12-final-v1.json")
CHUNK = "chk_" + "a" * 64


def _question(
    question_id: str = "001", split: str = "development"
) -> EvaluationQuestion:
    prefix = {"development": "dev", "validation": "val", "test": "test"}[split]
    answerable = question_id != "002"
    return EvaluationQuestion.from_dict(
        {
            "schema_version": 1,
            "question_id": f"s4-{prefix}-{question_id}",
            "question": "What does WAL do?",
            "requested_version": "18.6",
            "answerable": answerable,
            "answer_key_points": ["It records changes."] if answerable else [],
            "relevant_source_refs": (
                [
                    {
                        "version": "18.6",
                        "normalized_url": "https://www.postgresql.org/docs/18/wal.html",
                        "anchor": "WAL-INTRO",
                    }
                ]
                if answerable
                else []
            ),
            "resolved_chunk_ids": [CHUNK] if answerable else [],
            "chunking_version": "s2-chunk-v1",
            "expected_reason_code": None if answerable else "NO_EVIDENCE",
            "category": "single_fact" if answerable else "no_evidence_false_premise",
            "split": split,
            "source": "assistant_curated",
            "review_status": "source_verified",
            "paraphrase_group_id": f"wal-{split}-{question_id}",
        }
    )


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="E01",
        chunk_id=CHUNK,
        rank=1,
        score=0.9,
        postgresql_version="18.6",
        topic="wal",
        section_path=("WAL",),
        canonical_url="https://www.postgresql.org/docs/18/wal.html",
        anchor="WAL-INTRO",
        text="WAL records changes.",
        original_text_chars=20,
        text_truncated=False,
        provenance=EvidenceProvenance(
            sources=("dense",), dense_rank=1, dense_score=0.9
        ),
    )


def _answer() -> GeneratedAnswer:
    evidence = _evidence()
    citation = VerifiedCitation(
        evidence_id=evidence.evidence_id,
        chunk_id=evidence.chunk_id,
        postgresql_version=evidence.postgresql_version,
        topic=evidence.topic,
        section_path=evidence.section_path,
        canonical_url=evidence.canonical_url,
        anchor=evidence.anchor,
        text=evidence.text,
        score=evidence.score,
    )
    return GeneratedAnswer(
        status="success",
        answer="WAL records changes.",
        citations=(citation,),
        evidence=(evidence,),
        validation_status="verified",
        raw_content='{"answer":"WAL records changes.","citations":["E01"]}',
        request_id="req-1",
        model="gpt-5.4-mini-2026-03-17",
        usage=GenerationUsage(
            100,
            20,
            10,
            0.0001125,
            api_cost_usd_min=0.0001125,
            api_cost_usd_max=0.0001125,
        ),
        timing_ms=GenerationTiming(1.0, 20.0, 2.0, 28.0, retrieval=5.0),
        provider_fingerprint="fp-unit",
    )


class FakePipeline:
    def __init__(self, fail_once: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_once = fail_once

    async def answer(self, question, version, **switches):
        self.calls.append(question)
        assert switches == {
            "generation_enabled": True,
            "citation_validation_enabled": True,
        }
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("controlled secret=do-not-save")
        return _answer()


class _FixedClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _DenseWithOneHit:
    async def search(
        self, query: str, version: str | None, top_k: int, dense_enabled: bool
    ) -> tuple[DenseSearchHit, ...]:
        assert (query, version, top_k, dense_enabled) == (
            "What does WAL do?",
            "18.6",
            10,
            True,
        )
        return (
            DenseSearchHit(
                rank=1,
                chunk_id=CHUNK,
                score=0.9,
                postgresql_version="18.6",
                topic="wal",
                section_path=("WAL",),
                canonical_url="https://www.postgresql.org/docs/18/wal.html",
                anchor="WAL-INTRO",
                text="WAL records changes.",
            ),
        )

    async def close(self) -> None:
        return None


class _PreResponseFailureClient:
    async def generate(self, request) -> None:
        raise GenerationUpstreamError("transport failed api_key=do-not-save")

    async def close(self) -> None:
        return None


def _metadata(split: str = "development") -> dict[str, object]:
    config = load_generation_config(CONFIG_PATH)
    return {
        "schema_version": 1,
        "run_id": "s7-unit-b2",
        "git_commit": "f" * 40,
        "split": split,
        "dataset_version": "s4-golden-v1",
        "dataset_sha256": "1" * 64,
        "generation_config_sha256": "2" * 64,
        "generation_version": config.generation_version,
        "provider": config.provider,
        "protocol": config.protocol,
        "model": config.model,
        "documented_model_version": config.documented_model_version,
        "model_version_checked_on": config.model_version_checked_on,
        "prompt_version": config.prompt_version,
        "prompt_sha256": config.prompt_sha256,
        "output_schema_sha256": config.output_schema_sha256,
        "base_url": config.default_base_url,
        "minimum_input_cost_per_million_usd": config.minimum_input_cost_per_million_usd,
        "input_cost_per_million_usd": config.input_cost_per_million_usd,
        "minimum_cached_input_cost_per_million_usd": config.minimum_cached_input_cost_per_million_usd,
        "cached_input_cost_per_million_usd": config.cached_input_cost_per_million_usd,
        "minimum_output_cost_per_million_usd": config.minimum_output_cost_per_million_usd,
        "output_cost_per_million_usd": config.output_cost_per_million_usd,
        "pricing_checked_on": config.pricing_checked_on,
        "route": "b2",
        "parameters": {
            "dense_enabled": True,
            "bm25_enabled": False,
            "rerank_enabled": False,
            "generation_enabled": True,
            "citation_validation_enabled": True,
            "top_k": 10,
        },
    }


def test_runner_writes_complete_rows_summary_and_resumes(tmp_path: Path) -> None:
    pipeline = FakePipeline()
    questions = [_question("001"), _question("002")]
    output = tmp_path / "run"
    summary = asyncio.run(
        run_generation_evaluation(
            questions,
            output,
            pipeline,
            load_generation_config(CONFIG_PATH),
            _metadata(),
        )
    )
    assert summary["generation"]["answerable"]["citation_recall"] == 1.0
    assert summary["operations"]["usage"]["cached_input_tokens"] == 40
    rows = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    assert rows[0]["evidence"][0]["provenance"]["sources"] == ["dense"]
    assert rows[0]["citations"][0]["canonical_url"].endswith("wal.html")
    assert rows[0]["provider_fingerprint"] == "fp-unit"
    expected_raw = _answer().raw_content
    assert expected_raw is not None
    assert rows[0]["raw_content_sha256"] == hashlib.sha256(
        expected_raw.encode("utf-8")
    ).hexdigest()
    assert rows[0]["raw_content_chars"] == len(expected_raw)
    assert rows[0]["raw_content_bytes"] == len(expected_raw.encode("utf-8"))
    assert rows[0]["usage"] == {
        "api_cost_usd": 0.0001125,
        "api_cost_usd_max": 0.0001125,
        "api_cost_usd_min": 0.0001125,
        "cached_input_tokens": 20,
        "input_tokens": 100,
        "output_tokens": 10,
    }
    assert summary["operations"]["usage"]["api_cost_usd_min"] == 0.000225
    assert summary["operations"]["usage"]["api_cost_usd_max"] == 0.000225
    assert rows[0]["latency_ms"] == {
        "end_to_end": 28.0,
        "evidence": 1.0,
        "generation": 20.0,
        "retrieval": 5.0,
        "validation": 2.0,
    }
    asyncio.run(
        run_generation_evaluation(
            questions,
            output,
            pipeline,
            load_generation_config(CONFIG_PATH),
            _metadata(),
        )
    )
    assert len(pipeline.calls) == 2


def test_runner_never_reruns_failed_rows_and_redacts_secrets(tmp_path: Path) -> None:
    pipeline = FakePipeline(fail_once=True)
    output = tmp_path / "run"
    args = (
        [_question()],
        output,
        pipeline,
        load_generation_config(CONFIG_PATH),
        _metadata(),
    )
    first = asyncio.run(run_generation_evaluation(*args))
    assert first["operations"]["failed_count"] == 1
    failure_view = json.loads((output / "failures.jsonl").read_text())
    failure = json.loads((output / "results.jsonl").read_text())
    assert "do-not-save" not in failure_view["error_message"]
    assert "do-not-save" not in failure["error_message"]
    assert failure["raw_content"] is None
    assert failure["raw_content_sha256"] is None
    assert failure["raw_content_chars"] is None
    assert failure["raw_content_bytes"] is None
    assert failure["evidence"] == []
    assert failure["request_id"] is None
    assert failure["model"] is None
    assert failure["provider_fingerprint"] is None
    assert failure["usage"] == {
        "api_cost_usd": 0.0,
        "api_cost_usd_max": 0.0,
        "api_cost_usd_min": 0.0,
        "cached_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    second = asyncio.run(run_generation_evaluation(*args))
    assert second["operations"]["failed_count"] == 1
    assert len(pipeline.calls) == 1


@pytest.mark.parametrize(
    "raw_content",
    [
        '{"answer":"Unsupported.","citations":["E02"]}',
        '{"answer":"secret=do-not-save Unsupported.","citations":["E02"]}',
    ],
)
def test_runner_preserves_post_response_validation_failure_context_and_cost(
    tmp_path: Path, raw_content: str
) -> None:
    config = load_generation_config(CONFIG_PATH)
    completion = GenerationCompletion(
        content=raw_content,
        request_id="req-failed-validation",
        model=config.model,
        finish_reason="stop",
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=10,
        provider_fingerprint="fp-failed-validation",
    )
    service = GenerationService(
        config,
        lambda: FixedGenerationClient((completion,)),
        clock=_FixedClock((2.0, 2.001, 2.011, 2.013)),
    )
    pipeline = B2Pipeline(
        _DenseWithOneHit(),
        service,
        config,
        clock=_FixedClock((1.0, 1.005)),
    )
    output = tmp_path / "post-response-failure"

    summary = asyncio.run(
        run_generation_evaluation(
            [_question()], output, pipeline, config, _metadata()
        )
    )

    failure = json.loads((output / "results.jsonl").read_text())
    expected_cost = ((80 * 0.75) + (20 * 0.075) + (10 * 4.5)) / 1_000_000
    assert failure["status"] == "failed"
    assert failure["error_type"] == "GenerationValidationError"
    assert failure["raw_content"] is None
    assert failure["raw_content_sha256"] == hashlib.sha256(
        raw_content.encode("utf-8")
    ).hexdigest()
    assert failure["raw_content_chars"] == len(raw_content)
    assert failure["raw_content_bytes"] == len(raw_content.encode("utf-8"))
    assert failure["citations"] == []
    assert failure["evidence"][0]["evidence_id"] == "E01"
    assert failure["request_id"] == "req-failed-validation"
    assert failure["model"] == config.model
    assert failure["provider_fingerprint"] == "fp-failed-validation"
    assert failure["usage"] == {
        "api_cost_usd": pytest.approx(expected_cost),
        "api_cost_usd_max": pytest.approx(expected_cost),
        "api_cost_usd_min": pytest.approx(expected_cost),
        "cached_input_tokens": 20,
        "input_tokens": 100,
        "output_tokens": 10,
    }
    assert failure["latency_ms"] == {
        "retrieval": pytest.approx(5.0),
        "evidence": pytest.approx(1.0),
        "generation": pytest.approx(10.0),
        "validation": pytest.approx(2.0),
        "end_to_end": pytest.approx(18.0),
    }
    assert summary["operations"]["failed_count"] == 1
    assert summary["operations"]["success_count"] == 0
    assert summary["operations"]["usage"]["api_cost_usd"] == pytest.approx(
        expected_cost
    )
    assert summary["operations"]["usage"]["api_cost_usd_min"] == pytest.approx(
        expected_cost
    )
    assert summary["operations"]["usage"]["api_cost_usd_max"] == pytest.approx(
        expected_cost
    )


@pytest.mark.parametrize(
    "secret_field", ["password", "access_token", "authorization", "client_secret"]
)
def test_runner_suppresses_schema_invalid_json_with_sensitive_unknown_fields(
    tmp_path: Path, secret_field: str
) -> None:
    config = load_generation_config(CONFIG_PATH)
    completion = GenerationCompletion(
        content=json.dumps(
            {
                "answer": "WAL records changes.",
                "citations": ["E01"],
                secret_field: "do-not-save",
            },
            separators=(",", ":"),
        ),
        request_id="req-invalid-schema",
        model=config.model,
        finish_reason="stop",
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=10,
        provider_fingerprint="fp-invalid-schema",
    )
    service = GenerationService(
        config,
        lambda: FixedGenerationClient((completion,)),
        clock=_FixedClock((2.0, 2.001, 2.011, 2.013)),
    )
    pipeline = B2Pipeline(
        _DenseWithOneHit(),
        service,
        config,
        clock=_FixedClock((1.0, 1.005)),
    )
    output = tmp_path / "invalid-schema"

    summary = asyncio.run(
        run_generation_evaluation(
            [_question()], output, pipeline, config, _metadata()
        )
    )

    failure = json.loads((output / "results.jsonl").read_text())
    assert failure["status"] == "failed"
    assert failure["error_type"] == "GenerationValidationError"
    assert failure["raw_content"] is None
    assert failure["raw_content_sha256"] == hashlib.sha256(
        completion.content.encode("utf-8")
    ).hexdigest()
    assert failure["raw_content_chars"] == len(completion.content)
    assert failure["raw_content_bytes"] == len(completion.content.encode("utf-8"))
    assert failure["evidence"][0]["evidence_id"] == "E01"
    assert failure["request_id"] == "req-invalid-schema"
    assert failure["model"] == config.model
    assert failure["provider_fingerprint"] == "fp-invalid-schema"
    assert failure["usage"]["input_tokens"] == 100
    assert failure["usage"]["api_cost_usd"] > 0.0
    assert set(failure["latency_ms"]) == {
        "retrieval",
        "evidence",
        "generation",
        "validation",
        "end_to_end",
    }
    assert summary["operations"]["usage"]["api_cost_usd"] > 0.0


@pytest.mark.parametrize(
    ("case_id", "raw_content"),
    [
        (
            "api_key",
            '{"api_key":"do-not-save"',
        ),
        (
            "secret",
            'prefix "secret":"do-not-save" suffix',
        ),
        (
            "bearer",
            '{"answer":"safe text","bearer":"do-not-save"',
        ),
        (
            "api-key-unclosed-value",
            '{"api_key":"do-not-save',
        ),
        (
            "secret-multiline-value",
            '{"secret":"do-not-\nsave"',
        ),
        (
            "escaped-quote-truncated",
            '{"api_key":"do-not-\\"save',
        ),
        (
            "unquoted-value",
            '{"bearer":do-not-save}',
        ),
        (
            "safe-malformed",
            "plain malformed safe text",
        ),
    ],
)
def test_runner_suppresses_malformed_completion_raw_and_preserves_audit_metadata(
    tmp_path: Path,
    case_id: str,
    raw_content: str,
) -> None:
    config = load_generation_config(CONFIG_PATH)
    completion = GenerationCompletion(
        content=raw_content,
        request_id=f"req-malformed-{case_id}",
        model=config.model,
        finish_reason="stop",
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=10,
    )
    service = GenerationService(
        config,
        lambda: FixedGenerationClient((completion,)),
        clock=_FixedClock((2.0, 2.001, 2.011, 2.013)),
    )
    pipeline = B2Pipeline(
        _DenseWithOneHit(),
        service,
        config,
        clock=_FixedClock((1.0, 1.005)),
    )
    output = tmp_path / f"malformed-{case_id}"

    asyncio.run(
        run_generation_evaluation(
            [_question()], output, pipeline, config, _metadata()
        )
    )

    failure = json.loads((output / "results.jsonl").read_text())
    assert failure["error_type"] == "GenerationValidationError"
    assert failure["raw_content"] is None
    assert failure["raw_content_sha256"] == hashlib.sha256(
        raw_content.encode("utf-8")
    ).hexdigest()
    assert failure["raw_content_chars"] == len(raw_content)
    assert failure["raw_content_bytes"] == len(raw_content.encode("utf-8"))
    assert failure["usage"]["api_cost_usd"] > 0.0


def test_runner_keeps_typed_pre_response_failure_zero_cost_and_redacted(
    tmp_path: Path,
) -> None:
    config = load_generation_config(CONFIG_PATH)
    service = GenerationService(config, _PreResponseFailureClient)
    pipeline = B2Pipeline(_DenseWithOneHit(), service, config)
    output = tmp_path / "pre-response-failure"

    summary = asyncio.run(
        run_generation_evaluation(
            [_question()], output, pipeline, config, _metadata()
        )
    )

    failure = json.loads((output / "results.jsonl").read_text())
    assert failure["status"] == "failed"
    assert failure["error_type"] == "GenerationUpstreamError"
    assert "do-not-save" not in failure["error_message"]
    assert failure["raw_content"] is None
    assert failure["raw_content_sha256"] is None
    assert failure["raw_content_chars"] is None
    assert failure["raw_content_bytes"] is None
    assert failure["evidence"] == []
    assert failure["request_id"] is None
    assert failure["model"] is None
    assert failure["provider_fingerprint"] is None
    assert failure["usage"] == {
        "api_cost_usd": 0.0,
        "api_cost_usd_max": 0.0,
        "api_cost_usd_min": 0.0,
        "cached_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    assert summary["operations"]["usage"]["api_cost_usd"] == 0.0


def test_runner_rejects_test_and_changed_contract(tmp_path: Path) -> None:
    pipeline = FakePipeline()
    with pytest.raises(TestSplitLockedError, match="S12"):
        asyncio.run(
            run_generation_evaluation(
                [_question(split="test")],
                tmp_path / "test",
                pipeline,
                load_generation_config(CONFIG_PATH),
                _metadata("test"),
            )
        )
    assert pipeline.calls == []
    output = tmp_path / "development"
    asyncio.run(
        run_generation_evaluation(
            [_question()],
            output,
            pipeline,
            load_generation_config(CONFIG_PATH),
            _metadata(),
        )
    )
    changed = _metadata()
    changed["model"] = "rolling-alias"
    with pytest.raises(RunContractError, match="model"):
        asyncio.run(
            run_generation_evaluation(
                [_question()],
                output,
                pipeline,
                load_generation_config(CONFIG_PATH),
                changed,
            )
        )


def test_runner_accepts_only_complete_authorized_test_and_never_repeats(
    tmp_path: Path,
) -> None:
    test_path = tmp_path / S12_PROTOCOL.dataset.test_path
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes((ROOT / S12_PROTOCOL.dataset.test_path).read_bytes())
    authorization = authorize_test_run(tmp_path, S12_PROTOCOL, "b2")
    metadata = _metadata("test")
    metadata["run_id"] = authorization.run_id
    metadata["dataset_sha256"] = authorization.test_sha256
    questions = load_questions(ROOT / S12_PROTOCOL.dataset.test_path)
    pipeline = FakePipeline()

    first = asyncio.run(
        run_generation_evaluation(
            questions,
            authorization.output_dir,
            pipeline,
            load_generation_config(CONFIG_PATH),
            metadata,
            test_authorization=authorization,
        )
    )
    second = asyncio.run(
        run_generation_evaluation(
            questions,
            authorization.output_dir,
            pipeline,
            load_generation_config(CONFIG_PATH),
            metadata,
            test_authorization=authorization,
        )
    )

    assert first == second
    assert len(pipeline.calls) == 60
