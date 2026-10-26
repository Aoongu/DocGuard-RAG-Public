from pathlib import Path

import pytest

from docguard.generation.config import load_generation_config
from docguard.generation.models import (
    GeneratedAnswer,
    GenerationTiming,
    GenerationUsage,
)
from docguard.generation.pipeline import B2Pipeline
from docguard.retrieval.service import DenseSearchHit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_generation_config(
    PROJECT_ROOT / "configs" / "generation" / "s7-generation-v1.json"
)


def _answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        status="success",
        answer="Answer",
        citations=(),
        evidence=(),
        validation_status="structural_only",
        raw_content='{"answer":"Answer","citations":[]}',
        request_id="request",
        model=CONFIG.model,
        usage=GenerationUsage(1, 0, 1, 0.00001),
        timing_ms=GenerationTiming(0.0, 1.0, 1.0, 2.0, 0.0),
    )


class _Dense:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.closed = 0

    async def search(
        self, query: str, version: str | None, top_k: int, dense_enabled: bool
    ) -> list[DenseSearchHit]:
        self.calls.append((query, version, top_k, dense_enabled))
        return []

    async def close(self) -> None:
        self.closed += 1


class _Generation:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.closed = 0

    async def answer(
        self,
        question: str,
        requested_version: str | None,
        hits: tuple[DenseSearchHit, ...] | list[DenseSearchHit],
        *,
        generation_enabled: bool,
        citation_validation_enabled: bool,
    ) -> GeneratedAnswer:
        self.calls.append(
            (
                question,
                requested_version,
                tuple(hits),
                generation_enabled,
                citation_validation_enabled,
            )
        )
        return _answer()

    async def close(self) -> None:
        self.closed += 1


class _Clock:
    def __init__(self) -> None:
        self.values = iter((1.0, 1.025))

    def __call__(self) -> float:
        return next(self.values)


@pytest.mark.asyncio
async def test_b2_uses_only_dense_top10_and_propagates_switches() -> None:
    dense = _Dense()
    generation = _Generation()
    pipeline = B2Pipeline(dense, generation, CONFIG, clock=_Clock())

    result = await pipeline.answer(
        "What does wal_level control?",
        "17.11",
        generation_enabled=False,
        citation_validation_enabled=False,
    )

    assert dense.calls == [("What does wal_level control?", "17.11", 10, True)]
    assert generation.calls == [
        ("What does wal_level control?", "17.11", (), False, False)
    ]
    assert result.timing_ms.retrieval == pytest.approx(25.0)
    assert result.timing_ms.end_to_end == pytest.approx(27.0)
    await pipeline.close()
    await pipeline.close()
    assert dense.closed == 1
    assert generation.closed == 1


@pytest.mark.asyncio
async def test_b2_rejects_invalid_input_before_retrieval() -> None:
    dense = _Dense()
    generation = _Generation()
    pipeline = B2Pipeline(dense, generation, CONFIG)

    with pytest.raises(ValueError, match="question"):
        await pipeline.answer(" ", None)
    with pytest.raises(ValueError, match="version"):
        await pipeline.answer("Question", "19")
    with pytest.raises(TypeError, match="generation_enabled"):
        await pipeline.answer("Question", None, generation_enabled=1)
    assert dense.calls == []
    await pipeline.close()
