"""S7 B2 generation evaluation, summary, and human-review commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from docguard.evaluation.dataset import load_questions, sha256_file
from docguard.evaluation.final_protocol import S12TestAuthorization
from docguard.evaluation.generation_judge import (
    create_review_template,
    judge_output_schema_sha256,
    judge_prompt_sha256,
    run_judge_evaluation,
    select_review_sample,
)
from docguard.evaluation.generation_runner import run_generation_evaluation
from docguard.evaluation.models import EvaluationQuestion
from docguard.evaluation.runner import TestSplitLockedError, _atomic_text, _json_line
from docguard.fs_safety import validate_directory, validate_regular_file
from docguard.generation.client import (
    DeepSeekResponsesClient,
    FixedGenerationClient,
    OpenAICompatibleClient,
    StructuredGenerationClient,
)
from docguard.generation.config import (
    GenerationConfig,
    load_generation_config,
    normalize_base_url,
)
from docguard.generation.models import GenerationCompletion
from docguard.generation.pipeline import B2Pipeline
from docguard.generation.service import GenerationService
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.service import create_default_retriever

ClientFactory = Callable[[], StructuredGenerationClient]
_COMPLETION_FIELDS = frozenset(
    {
        "content",
        "request_id",
        "model",
        "finish_reason",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
    }
)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _load_mock_completions(
    path: Path, config: GenerationConfig
) -> tuple[GenerationCompletion, ...]:
    validate_regular_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("mock response file must be a non-empty JSON array")
    completions: list[GenerationCompletion] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != _COMPLETION_FIELDS:
            raise ValueError(f"invalid mock completion fields at index {index}")
        if item["model"] != config.model or item["finish_reason"] != "stop":
            raise ValueError(f"mock completion identity mismatch at index {index}")
        for field in ("input_tokens", "cached_input_tokens", "output_tokens"):
            if type(item[field]) is not int or item[field] < 0:
                raise ValueError(f"mock completion {field} must be non-negative")
        if item["cached_input_tokens"] > item["input_tokens"]:
            raise ValueError("mock cached input tokens exceed input tokens")
        if not isinstance(item["content"], str) or not isinstance(
            item["request_id"], str
        ):
            raise TypeError("mock completion text fields are invalid")
        completions.append(GenerationCompletion(**item))
    return tuple(completions)


def _build_generation_client_factory(
    config: GenerationConfig,
    mock_response_file: Path | None,
    environment: Mapping[str, str],
) -> ClientFactory:
    if mock_response_file is not None:
        completions = _load_mock_completions(mock_response_file, config)
        return lambda: FixedGenerationClient(completions)
    if config.protocol == "deepseek-responses":
        api_key = environment.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for a real S7 DeepSeek run")
        environment_model = environment.get("DEEPSEEK_MODEL", "").strip()
        if environment_model and environment_model != config.model:
            raise ValueError("DEEPSEEK_MODEL must match the frozen model alias")
        environment_base_url = environment.get("DEEPSEEK_BASE_URL", "").strip()
        if (
            environment_base_url
            and normalize_base_url(environment_base_url) != config.default_base_url
        ):
            raise ValueError("DEEPSEEK_BASE_URL must match the frozen config")
        return lambda: DeepSeekResponsesClient(
            config,
            api_key=api_key,
            base_url=environment_base_url or None,
        )
    api_key = environment.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for a real S7 run")
    environment_model = environment.get("OPENAI_MODEL", "").strip()
    if environment_model and environment_model != config.model:
        raise ValueError("OPENAI_MODEL must match the frozen model snapshot")
    environment_base_url = environment.get("OPENAI_BASE_URL", "").strip()
    if (
        environment_base_url
        and normalize_base_url(environment_base_url) != config.default_base_url
    ):
        raise ValueError("OPENAI_BASE_URL must match the frozen config")
    return lambda: OpenAICompatibleClient(
        config,
        api_key=api_key,
        base_url=environment_base_url or None,
    )


def _select_real_smoke_question(
    questions: Sequence[EvaluationQuestion], question_id: str
) -> EvaluationQuestion:
    selected = [item for item in questions if item.question_id == question_id]
    if len(selected) != 1:
        raise ValueError("real smoke question_id must select exactly one question")
    if selected[0].split != "development":
        raise ValueError("real smoke may only use a Development question")
    return selected[0]


def _validate_b2_local_assets(
    *,
    index_root: Path,
    model_root: Path,
    model_manifest: Path,
    tokenizer: Path,
) -> None:
    """Fail before creating run artifacts when required local assets are unavailable."""
    validate_directory(index_root / "qdrant")
    verify_model_assets(model_root, model_manifest)
    validate_regular_file(tokenizer)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/generation/s7-generation-v1.json"),
    )
    run = subparsers.add_parser("run-b2")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/generation/s7-generation-v1.json"),
    )
    smoke = subparsers.add_parser("run-real-smoke")
    smoke.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/s4-golden-v1/development.jsonl"),
    )
    smoke.add_argument("--question-id", required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--run-id", required=True)
    smoke.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/generation/s7-deepseek-flash-v1.json"),
    )
    smoke.add_argument("--index-root", type=Path, default=Path(".qdrant/s3-dense-v1"))
    smoke.add_argument(
        "--dense-config", type=Path, default=Path("configs/retrieval/s3-dense-v1.json")
    )
    smoke.add_argument(
        "--model-root", type=Path, default=Path("data/models/s3-e5-small")
    )
    smoke.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("data/manifests/s3-embedding-model.json"),
    )
    smoke.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json"
        ),
    )
    run.add_argument("--mock-response-file", type=Path)
    run.add_argument("--index-root", type=Path, default=Path(".qdrant/s3-dense-v1"))
    run.add_argument(
        "--dense-config", type=Path, default=Path("configs/retrieval/s3-dense-v1.json")
    )
    run.add_argument("--model-root", type=Path, default=Path("data/models/s3-e5-small"))
    run.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("data/manifests/s3-embedding-model.json"),
    )
    run.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json"
        ),
    )
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--summary", type=Path, required=True)
    review = subparsers.add_parser("create-review-template")
    review.add_argument("--dataset", type=Path, required=True)
    review.add_argument("--generation-results", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--per-stratum", type=int, default=2)
    judge = subparsers.add_parser("run-judge")
    judge.add_argument("--generation-results", type=Path, required=True)
    judge.add_argument("--human-review", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--run-id", required=True)
    judge.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/generation/s7-deepseek-flash-v1.json"),
    )
    judge.add_argument("--mock-response-file", type=Path)
    return parser


async def _run_b2(
    args: argparse.Namespace,
    *,
    test_authorization: S12TestAuthorization | None = None,
) -> dict[str, object]:
    questions = load_questions(args.dataset)
    smoke_only = args.command == "run-real-smoke"
    if smoke_only:
        questions = [_select_real_smoke_question(questions, args.question_id)]
    splits = {question.split for question in questions}
    if splits == {"test"} and test_authorization is None:
        raise TestSplitLockedError("Test split is frozen and cannot run before S12")
    if len(splits) != 1:
        raise ValueError("one B2 run may contain exactly one split")
    config = load_generation_config(args.generation_config)
    client_factory = _build_generation_client_factory(
        config, getattr(args, "mock_response_file", None), os.environ
    )
    _validate_b2_local_assets(
        index_root=args.index_root,
        model_root=args.model_root,
        model_manifest=args.model_manifest,
        tokenizer=args.tokenizer,
    )
    dense = create_default_retriever(
        index_root=args.index_root,
        config_path=args.dense_config,
        model_root=args.model_root,
        model_manifest_path=args.model_manifest,
        tokenizer_path=args.tokenizer,
    )
    service = GenerationService(config, client_factory)
    pipeline = B2Pipeline(dense, service, config)
    split = next(iter(splits))
    metadata: dict[str, object] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "git_commit": _git_commit(),
        "split": split,
        "dataset_version": config.dataset_version,
        "dataset_sha256": sha256_file(args.dataset),
        "generation_config_sha256": sha256_file(args.generation_config),
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
        "smoke_only": smoke_only,
        "route": "b2",
        "parameters": {
            "dense_enabled": True,
            "bm25_enabled": False,
            "rerank_enabled": False,
            "generation_enabled": True,
            "citation_validation_enabled": True,
            "top_k": config.evidence_top_k,
        },
    }
    try:
        summary = await run_generation_evaluation(
            questions,
            args.output,
            pipeline,
            config,
            metadata,
            test_authorization=test_authorization,
        )
        if smoke_only and summary["operations"]["failed_count"] != 0:
            raise RuntimeError(
                "real DeepSeek smoke failed; no automatic retry is allowed"
            )
        return summary
    finally:
        await pipeline.close()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    validate_regular_file(path)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError("JSONL input must not be empty")
    return rows


async def _run_judge(args: argparse.Namespace) -> dict[str, object]:
    generation_rows = _read_jsonl(args.generation_results)
    validate_regular_file(args.human_review)
    human_review = json.loads(args.human_review.read_text(encoding="utf-8"))
    if not isinstance(human_review, dict):
        raise TypeError("human review must be a JSON object")
    config = load_generation_config(args.generation_config)
    factory = _build_generation_client_factory(
        config, args.mock_response_file, os.environ
    )
    generated_models = {
        row.get("model")
        for row in generation_rows
        if row.get("status") == "success" and isinstance(row.get("model"), str)
    }
    same_model = not generated_models or generated_models == {config.model}
    metadata: dict[str, object] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "git_commit": _git_commit(),
        "split": "development",
        "judge_prompt_version": "s7-judge-v1",
        "judge_prompt_sha256": judge_prompt_sha256(),
        "judge_output_schema_sha256": judge_output_schema_sha256(),
        "generation_results_sha256": sha256_file(args.generation_results),
        "human_review_sha256": sha256_file(args.human_review),
        "generation_config_sha256": sha256_file(args.generation_config),
        "provider": config.provider,
        "protocol": config.protocol,
        "model": config.model,
        "documented_model_version": config.documented_model_version,
        "model_version_checked_on": config.model_version_checked_on,
        "base_url": config.default_base_url,
        "pricing_checked_on": config.pricing_checked_on,
        "minimum_input_cost_per_million_usd": (
            config.minimum_input_cost_per_million_usd
        ),
        "input_cost_per_million_usd": config.input_cost_per_million_usd,
        "minimum_cached_input_cost_per_million_usd": (
            config.minimum_cached_input_cost_per_million_usd
        ),
        "cached_input_cost_per_million_usd": (config.cached_input_cost_per_million_usd),
        "minimum_output_cost_per_million_usd": (
            config.minimum_output_cost_per_million_usd
        ),
        "output_cost_per_million_usd": config.output_cost_per_million_usd,
        "same_model_as_generator": same_model,
        "independent_prompt_from_generator": True,
        "max_retries": 0,
    }
    client = factory()
    try:
        return await run_judge_evaluation(
            generation_rows,
            human_review,
            args.output,
            client,
            config,
            metadata,
        )
    finally:
        await client.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_generation_config(args.generation_config)
            result: dict[str, object] = {
                "status": "valid",
                "generation_version": config.generation_version,
                "model": config.model,
                "prompt_sha256": config.prompt_sha256,
                "output_schema_sha256": config.output_schema_sha256,
            }
        elif args.command in {"run-b2", "run-real-smoke"}:
            result = asyncio.run(_run_b2(args))
        elif args.command == "run-judge":
            result = asyncio.run(_run_judge(args))
        elif args.command == "summarize":
            validate_regular_file(args.summary)
            result = json.loads(args.summary.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise TypeError("summary must be a JSON object")
        else:
            questions = load_questions(args.dataset)
            sample = select_review_sample(questions, per_stratum=args.per_stratum)
            validate_regular_file(args.generation_results)
            generation_rows = [
                json.loads(line)
                for line in args.generation_results.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            sample_ids = {question.question_id for question in sample}
            sample_results = [
                row
                for row in generation_rows
                if isinstance(row, dict) and row.get("question_id") in sample_ids
            ]
            result = create_review_template(sample, sample_results)
            _atomic_text(args.output, _json_line(result))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
