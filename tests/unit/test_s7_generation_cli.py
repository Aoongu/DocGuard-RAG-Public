import json
from pathlib import Path

import pytest

from docguard.evaluation.generation_cli import (
    _build_generation_client_factory,
    _select_real_smoke_question,
    _validate_b2_local_assets,
    main,
)
from docguard.evaluation.models import EvaluationQuestion
from docguard.fs_safety import FileSafetyError
from docguard.generation.client import DeepSeekResponsesClient, FixedGenerationClient
from docguard.generation.config import load_generation_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/generation/s7-generation-v1.json"
DEEPSEEK_CONFIG = ROOT / "configs/generation/s7-deepseek-flash-v1.json"


def test_validate_config_command(capsys) -> None:
    assert main(["validate-config", "--generation-config", str(CONFIG)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["generation_version"] == "s7-generation-v1"
    assert result["model"] == "gpt-5.4-mini-2026-03-17"


def test_real_client_requires_credentials_before_construction() -> None:
    config = load_generation_config(CONFIG)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _build_generation_client_factory(config, None, {})


def test_real_environment_overrides_must_match_frozen_config() -> None:
    config = load_generation_config(CONFIG)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        _build_generation_client_factory(
            config,
            None,
            {"OPENAI_API_KEY": "secret", "OPENAI_MODEL": "rolling-alias"},
        )
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        _build_generation_client_factory(
            config,
            None,
            {"OPENAI_API_KEY": "secret", "OPENAI_BASE_URL": "https://example.com/v1"},
        )


def test_deepseek_real_client_uses_provider_specific_credentials() -> None:
    config = load_generation_config(DEEPSEEK_CONFIG)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _build_generation_client_factory(config, None, {"OPENAI_API_KEY": "wrong-key"})

    factory = _build_generation_client_factory(
        config,
        None,
        {
            "DEEPSEEK_API_KEY": "secret",
            "DEEPSEEK_MODEL": "deepseek-flash",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
        },
    )
    assert isinstance(factory(), DeepSeekResponsesClient)


def test_deepseek_environment_overrides_must_match_frozen_config() -> None:
    config = load_generation_config(DEEPSEEK_CONFIG)
    with pytest.raises(ValueError, match="DEEPSEEK_MODEL"):
        _build_generation_client_factory(
            config,
            None,
            {"DEEPSEEK_API_KEY": "secret", "DEEPSEEK_MODEL": "deepseek-v4-pro"},
        )
    with pytest.raises(ValueError, match="DEEPSEEK_BASE_URL"):
        _build_generation_client_factory(
            config,
            None,
            {"DEEPSEEK_API_KEY": "secret", "DEEPSEEK_BASE_URL": "https://example.com"},
        )


def _question(question_id: str, split: str = "development") -> EvaluationQuestion:
    return EvaluationQuestion.from_dict(
        {
            "schema_version": 1,
            "question_id": question_id,
            "question": "Question",
            "requested_version": None,
            "answerable": False,
            "answer_key_points": [],
            "relevant_source_refs": [],
            "resolved_chunk_ids": [],
            "chunking_version": "s2-chunk-v1",
            "expected_reason_code": "NO_EVIDENCE",
            "category": "no_evidence_false_premise",
            "split": split,
            "source": "assistant_curated",
            "review_status": "human_reviewed",
            "paraphrase_group_id": question_id,
        }
    )


def test_real_smoke_selects_exactly_one_development_question() -> None:
    questions = [_question("s4-dev-001"), _question("s4-dev-002")]
    assert _select_real_smoke_question(questions, "s4-dev-002").question_id == (
        "s4-dev-002"
    )
    with pytest.raises(ValueError, match="exactly one"):
        _select_real_smoke_question(questions, "missing")
    with pytest.raises(ValueError, match="Development"):
        _select_real_smoke_question([_question("s4-test-001", "test")], "s4-test-001")


def test_b2_local_asset_preflight_rejects_missing_index_before_run(tmp_path: Path) -> None:
    with pytest.raises(FileSafetyError, match="unable to inspect directory"):
        _validate_b2_local_assets(
            index_root=tmp_path / "missing-index",
            model_root=tmp_path / "unused-model",
            model_manifest=tmp_path / "unused-manifest.json",
            tokenizer=tmp_path / "unused-tokenizer.json",
        )


def test_mock_response_file_builds_fixed_client(tmp_path: Path) -> None:
    response = tmp_path / "mock.json"
    response.write_text(
        json.dumps(
            [
                {
                    "content": '{"answer":"A","citations":["E01"]}',
                    "request_id": "mock-1",
                    "model": "gpt-5.4-mini-2026-03-17",
                    "finish_reason": "stop",
                    "input_tokens": 10,
                    "cached_input_tokens": 2,
                    "output_tokens": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    factory = _build_generation_client_factory(
        load_generation_config(CONFIG), response, {}
    )
    assert isinstance(factory(), FixedGenerationClient)


def test_summarize_command_reads_saved_summary(tmp_path: Path, capsys) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"ok"}\n', encoding="utf-8")
    assert main(["summarize", "--summary", str(summary)]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


def test_create_review_template_uses_frozen_generation_results(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "template.json"
    assert (
        main(
            [
                "create-review-template",
                "--dataset",
                str(ROOT / "data/eval/s4-golden-v1/development.jsonl"),
                "--generation-results",
                str(ROOT / "reports/s7_generation/development_b2/results.jsonl"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    rendered = json.loads(output.read_text(encoding="utf-8"))
    assert len(rendered["rows"]) == 16
    failed = next(
        row for row in rendered["rows"] if row["question_id"] == "s4-dev-032"
    )
    assert failed["review_status"] == "non_evaluable"
    assert failed["failure_reason_codes"] == ["generation_contract_failure"]
    assert json.loads(capsys.readouterr().out)["schema_version"] == 2


def test_run_judge_command_supports_frozen_mock_execution(
    tmp_path: Path, capsys
) -> None:
    generation_results = tmp_path / "generation.jsonl"
    generation_results.write_text(
        json.dumps(
            {
                "question_id": "s4-dev-001",
                "question": "What is supported?",
                "split": "development",
                "status": "success",
                "answer": "Supported answer.",
                "citations": [
                    {
                        "evidence_id": "E01",
                        "chunk_id": "chk_" + "1" * 64,
                        "postgresql_version": "18.6",
                        "topic": "sql_commands",
                        "section_path": ["Description"],
                        "canonical_url": "https://www.postgresql.org/docs/18/sql-select.html",
                        "anchor": "description",
                        "text": "Supported answer.",
                        "score": 0.9,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    human_review = tmp_path / "human.json"
    human_review.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "human_validation_status": "complete",
                "judge_prompt_version": "s7-judge-v1",
                "rows": [
                    {
                        "question_id": "s4-dev-001",
                        "generation_status": "success",
                        "visible_answer": "Supported answer.",
                        "review_status": "evaluable",
                        "human_faithfulness_score": 2,
                        "human_answer_relevance_score": 2,
                        "primary_failure_reason": None,
                        "failure_reason_codes": [],
                        "failure_reason_notes": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    mock = tmp_path / "mock.json"
    mock.write_text(
        json.dumps(
            [
                {
                    "content": json.dumps(
                        {
                            "faithfulness_score": 2,
                            "answer_relevance_score": 2,
                            "rationale": "Supported.",
                        }
                    ),
                    "request_id": "judge-1",
                    "model": "deepseek-flash",
                    "finish_reason": "stop",
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "output_tokens": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "judge"
    assert (
        main(
            [
                "run-judge",
                "--generation-results",
                str(generation_results),
                "--human-review",
                str(human_review),
                "--output",
                str(output),
                "--run-id",
                "judge-mock",
                "--generation-config",
                str(DEEPSEEK_CONFIG),
                "--mock-response-file",
                str(mock),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["agreement"]["judge_human_intersection_count"] == 1
    assert json.loads((output / "run.json").read_text())["same_model_as_generator"]
