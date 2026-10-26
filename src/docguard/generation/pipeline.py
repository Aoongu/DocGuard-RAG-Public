"""B2 composition: frozen Dense Top-10 followed by structured generation."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Protocol

from docguard.generation.config import GenerationConfig
from docguard.generation.models import GeneratedAnswer
from docguard.generation.service import PostResponseGenerationError

_VERSIONS = frozenset({"17.11", "18.6"})


class DenseRoute(Protocol):
    async def search(
        self, query: str, version: str | None, top_k: int, dense_enabled: bool
    ) -> Sequence[Any]: ...

    async def close(self) -> None: ...


class AnswerGenerator(Protocol):
    async def answer(
        self,
        question: str,
        requested_version: str | None,
        hits: Sequence[Any],
        *,
        generation_enabled: bool,
        citation_validation_enabled: bool,
    ) -> GeneratedAnswer: ...

    async def close(self) -> None: ...


class B2Pipeline:
    """Keep the Dense + LLM baseline isolated from M1/M2 and S8 logic."""

    def __init__(
        self,
        dense: DenseRoute,
        generation: AnswerGenerator,
        config: GenerationConfig,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._dense = dense
        self._generation = generation
        self._config = config
        self._clock = clock
        self._closed = False

    async def answer(
        self,
        question: str,
        requested_version: str | None,
        *,
        generation_enabled: bool = True,
        citation_validation_enabled: bool = True,
    ) -> GeneratedAnswer:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if requested_version is not None and requested_version not in _VERSIONS:
            raise ValueError("version must be null, 17.11, or 18.6")
        if type(generation_enabled) is not bool:
            raise TypeError("generation_enabled must be boolean")
        if type(citation_validation_enabled) is not bool:
            raise TypeError("citation_validation_enabled must be boolean")
        started = self._clock()
        hits = await self._dense.search(
            question.strip(),
            requested_version,
            self._config.evidence_top_k,
            dense_enabled=True,
        )
        retrieved = self._clock()
        retrieval_ms = (retrieved - started) * 1000.0
        try:
            generated = await self._generation.answer(
                question.strip(),
                requested_version,
                hits,
                generation_enabled=generation_enabled,
                citation_validation_enabled=citation_validation_enabled,
            )
        except PostResponseGenerationError as error:
            timing = replace(
                error.context.timing_ms,
                retrieval=retrieval_ms,
                end_to_end=retrieval_ms + error.context.timing_ms.end_to_end,
            )
            context = replace(error.context, timing_ms=timing)
            raise PostResponseGenerationError(
                error.validation_error, context
            ) from error.validation_error
        timing = replace(
            generated.timing_ms,
            retrieval=retrieval_ms,
            end_to_end=retrieval_ms + generated.timing_ms.end_to_end,
        )
        return replace(generated, timing_ms=timing)

    async def close(self) -> None:
        if not self._closed:
            await self._generation.close()
            await self._dense.close()
            self._closed = True
