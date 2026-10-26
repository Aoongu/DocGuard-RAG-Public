import asyncio
import json
from pathlib import Path

from docguard.evaluation.generation_runner import run_generation_evaluation
from docguard.evaluation.models import EvaluationQuestion
from docguard.generation.client import FixedGenerationClient
from docguard.generation.config import load_generation_config
from docguard.generation.models import GenerationCompletion
from docguard.generation.pipeline import B2Pipeline
from docguard.generation.service import GenerationService
from docguard.retrieval.service import DenseSearchHit

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/generation/s7-deepseek-flash-v1.json"
CHUNK = "chk_" + "b" * 64


class FixedDense:
    async def search(self, query, version, top_k, dense_enabled):
        assert (query, version, top_k, dense_enabled) == (
            "What does WAL do?",
            "18.6",
            10,
            True,
        )
        return [
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
            )
        ]

    async def close(self):
        return None


def _question() -> EvaluationQuestion:
    return EvaluationQuestion.from_dict(
        {
            "schema_version": 1,
            "question_id": "s4-dev-001",
            "question": "What does WAL do?",
            "requested_version": "18.6",
            "answerable": True,
            "answer_key_points": ["It records changes."],
            "relevant_source_refs": [
                {
                    "version": "18.6",
                    "normalized_url": "https://www.postgresql.org/docs/18/wal.html",
                    "anchor": "WAL-INTRO",
                }
            ],
            "resolved_chunk_ids": [CHUNK],
            "chunking_version": "s2-chunk-v1",
            "expected_reason_code": None,
            "category": "single_fact",
            "split": "development",
            "source": "human_written",
            "review_status": "human_reviewed",
            "paraphrase_group_id": "deepseek-smoke",
        }
    )


def _metadata(config) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "s7-deepseek-mock-smoke",
        "git_commit": "f" * 40,
        "split": "development",
        "dataset_version": config.dataset_version,
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


def test_deepseek_mock_smoke_saves_cost_bounds_and_optional_identity(
    tmp_path: Path,
) -> None:
    config = load_generation_config(CONFIG_PATH)
    completion = GenerationCompletion(
        content='{"answer":"WAL records changes.","citations":["E01"]}',
        request_id="resp-mock-1",
        model=config.model,
        finish_reason="completed",
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=10,
        provider_fingerprint=None,
    )
    values = iter([0.0, 0.001, 0.003, 0.004])
    service = GenerationService(
        config,
        lambda: FixedGenerationClient([completion]),
        clock=lambda: next(values),
    )
    pipeline = B2Pipeline(FixedDense(), service, config, clock=lambda: 0.0)
    try:
        summary = asyncio.run(
            run_generation_evaluation(
                [_question()], tmp_path, pipeline, config, _metadata(config)
            )
        )
    finally:
        asyncio.run(pipeline.close())

    row = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert summary["operations"]["success_count"] == 1
    assert row["provider_fingerprint"] is None
    assert 0 < row["usage"]["api_cost_usd_min"]
    assert row["usage"]["api_cost_usd_min"] <= row["usage"]["api_cost_usd_max"]
    assert row["usage"]["api_cost_usd"] == row["usage"]["api_cost_usd_max"]
    assert row["validation_status"] == "verified"
