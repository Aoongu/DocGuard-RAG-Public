from pathlib import Path

import pytest

from docguard.generation.client import FixedGenerationClient, GenerationResponseError
from docguard.generation.config import load_generation_config
from docguard.generation.models import GenerationCompletion
from docguard.generation.service import GenerationService
from docguard.retrieval.service import DenseSearchHit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_generation_config(
    PROJECT_ROOT / "configs" / "generation" / "s7-generation-v1.json"
)
DEEPSEEK_CONFIG = load_generation_config(
    PROJECT_ROOT / "configs" / "generation" / "s7-deepseek-flash-v1.json"
)
CHUNK = "chk_" + "a" * 64


def _hit() -> DenseSearchHit:
    return DenseSearchHit(
        rank=1,
        chunk_id=CHUNK,
        score=0.9,
        postgresql_version="17.11",
        topic="runtime-config",
        section_path=("Server Configuration", "Write Ahead Log"),
        canonical_url="https://www.postgresql.org/docs/17/runtime-config-wal.html",
        anchor="RUNTIME-CONFIG-WAL-SETTINGS",
        text="wal_level controls the amount of information written to WAL.",
    )


def _completion(content: str | None = None) -> GenerationCompletion:
    return GenerationCompletion(
        content=content
        or '{"answer":"wal_level controls WAL detail.","citations":["E01"]}',
        request_id="chatcmpl-fixed",
        model=CONFIG.model,
        finish_reason="stop",
        input_tokens=120,
        cached_input_tokens=20,
        output_tokens=18,
    )


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        value = self.value
        self.value += 0.010
        return value


@pytest.mark.asyncio
async def test_generation_service_builds_prompt_verifies_answer_and_computes_cost() -> (
    None
):
    client = FixedGenerationClient((_completion(),))
    created = 0

    def factory() -> FixedGenerationClient:
        nonlocal created
        created += 1
        return client

    service = GenerationService(CONFIG, factory, clock=_Clock())
    result = await service.answer(
        "What does wal_level control?",
        "17.11",
        (_hit(),),
        generation_enabled=True,
        citation_validation_enabled=True,
    )

    assert created == 1
    assert result.status == "success"
    assert result.answer == "wal_level controls WAL detail."
    assert result.raw_content == _completion().content
    assert result.validation_status == "verified"
    assert result.citations[0].chunk_id == CHUNK
    assert result.evidence[0].evidence_id == "E01"
    assert result.request_id == "chatcmpl-fixed"
    assert result.model == CONFIG.model
    assert result.usage.input_tokens == 120
    assert result.usage.cached_input_tokens == 20
    assert result.usage.output_tokens == 18
    expected_cost = ((100 * 0.75) + (20 * 0.075) + (18 * 4.5)) / 1_000_000
    assert result.usage.api_cost_usd == pytest.approx(expected_cost)
    assert result.usage.api_cost_usd_min == pytest.approx(expected_cost)
    assert result.usage.api_cost_usd_max == pytest.approx(expected_cost)
    assert result.timing_ms.evidence == 10.0
    assert result.timing_ms.generation == 10.0
    assert result.timing_ms.validation == pytest.approx(10.0)
    assert result.timing_ms.end_to_end == pytest.approx(30.0)
    await service.close()
    await service.close()


@pytest.mark.asyncio
async def test_deepseek_service_computes_cost_range_and_preserves_fingerprint() -> None:
    completion = GenerationCompletion(
        content='{"answer":"wal_level controls WAL detail.","citations":["E01"]}',
        request_id="resp-fixed",
        model=DEEPSEEK_CONFIG.model,
        finish_reason="completed",
        input_tokens=120,
        cached_input_tokens=20,
        output_tokens=18,
        provider_fingerprint="fp_test_1017",
    )
    service = GenerationService(
        DEEPSEEK_CONFIG, lambda: FixedGenerationClient((completion,)), clock=_Clock()
    )
    result = await service.answer(
        "What does wal_level control?",
        "17.11",
        (_hit(),),
        generation_enabled=True,
        citation_validation_enabled=True,
    )

    minimum = ((100 * 0.15) + (20 * 0.003) + (18 * 0.6)) / 1_000_000
    maximum = ((100 * 0.3) + (20 * 0.006) + (18 * 1.2)) / 1_000_000
    assert result.usage.api_cost_usd_min == pytest.approx(minimum)
    assert result.usage.api_cost_usd_max == pytest.approx(maximum)
    assert result.usage.api_cost_usd == pytest.approx(maximum)
    assert result.provider_fingerprint == "fp_test_1017"
    await service.close()


@pytest.mark.asyncio
async def test_generation_disabled_maps_evidence_without_constructing_client() -> None:
    def forbidden_factory() -> FixedGenerationClient:
        raise AssertionError("generation client must stay lazy")

    service = GenerationService(CONFIG, forbidden_factory, clock=_Clock())
    result = await service.answer(
        "What does wal_level control?",
        "17.11",
        (_hit(),),
        generation_enabled=False,
        citation_validation_enabled=True,
    )

    assert result.status == "disabled"
    assert result.answer is None
    assert result.citations == ()
    assert result.evidence[0].chunk_id == CHUNK
    assert result.request_id is None
    assert result.model is None
    assert result.usage.api_cost_usd == 0.0
    assert result.timing_ms.generation == 0.0
    assert result.timing_ms.validation == 0.0
    await service.close()


@pytest.mark.asyncio
async def test_generation_service_propagates_client_and_validation_errors() -> None:
    exhausted = GenerationService(CONFIG, lambda: FixedGenerationClient(()))
    with pytest.raises(GenerationResponseError, match="exhausted"):
        await exhausted.answer(
            "Question",
            None,
            (_hit(),),
            generation_enabled=True,
            citation_validation_enabled=True,
        )
    await exhausted.close()

    unknown = GenerationService(
        CONFIG,
        lambda: FixedGenerationClient(
            (_completion('{"answer":"Bad citation.","citations":["E02"]}'),)
        ),
    )
    with pytest.raises(RuntimeError, match="unknown evidence_id"):
        await unknown.answer(
            "Question",
            None,
            (_hit(),),
            generation_enabled=True,
            citation_validation_enabled=True,
        )
    await unknown.close()


@pytest.mark.asyncio
async def test_generation_service_structural_only_mode_can_return_no_citations() -> (
    None
):
    service = GenerationService(
        CONFIG,
        lambda: FixedGenerationClient(
            (_completion('{"answer":"Insufficient evidence.","citations":[]}'),)
        ),
    )
    result = await service.answer(
        "Question",
        None,
        (_hit(),),
        generation_enabled=True,
        citation_validation_enabled=False,
    )
    assert result.validation_status == "structural_only"
    assert result.citations == ()
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["generation_enabled", "citation_validation_enabled"])
async def test_generation_service_requires_boolean_switches(field: str) -> None:
    service = GenerationService(CONFIG, lambda: FixedGenerationClient((_completion(),)))
    kwargs = {
        "generation_enabled": True,
        "citation_validation_enabled": True,
    }
    kwargs[field] = 1
    with pytest.raises(TypeError, match=field):
        await service.answer("Question", None, (_hit(),), **kwargs)
    await service.close()
