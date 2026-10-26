"""Structured answer generation over already retrieved evidence."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from docguard.generation.citations import (
    GenerationValidationError,
    parse_generation_output,
    verify_citations,
)
from docguard.generation.client import StructuredGenerationClient
from docguard.generation.config import GenerationConfig
from docguard.generation.evidence import build_evidence
from docguard.generation.models import (
    GeneratedAnswer,
    GenerationFailureContext,
    GenerationRequest,
    GenerationTiming,
    GenerationUsage,
)
from docguard.generation.prompt import build_messages, output_schema


class PostResponseGenerationError(RuntimeError):
    """A local validation failure after a billable completion was received."""

    def __init__(
        self,
        validation_error: GenerationValidationError,
        context: GenerationFailureContext,
    ) -> None:
        super().__init__(str(validation_error))
        self.validation_error = validation_error
        self.context = context


class GenerationService:
    """Generate and verify one answer without owning retrieval."""

    def __init__(
        self,
        config: GenerationConfig,
        client_factory: Callable[[], StructuredGenerationClient],
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._client_factory = client_factory
        self._clock = clock
        self._client: StructuredGenerationClient | None = None

    def _get_client(self) -> StructuredGenerationClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _usage(
        self, input_tokens: int, cached: int, output_tokens: int
    ) -> GenerationUsage:
        uncached = input_tokens - cached
        minimum_cost = (
            uncached * self._config.minimum_input_cost_per_million_usd
            + cached * self._config.minimum_cached_input_cost_per_million_usd
            + output_tokens * self._config.minimum_output_cost_per_million_usd
        ) / 1_000_000
        maximum_cost = (
            uncached * self._config.input_cost_per_million_usd
            + cached * self._config.cached_input_cost_per_million_usd
            + output_tokens * self._config.output_cost_per_million_usd
        ) / 1_000_000
        return GenerationUsage(
            input_tokens,
            cached,
            output_tokens,
            maximum_cost,
            api_cost_usd_min=minimum_cost,
            api_cost_usd_max=maximum_cost,
        )

    async def answer(
        self,
        question: str,
        requested_version: str | None,
        hits: Sequence[Any],
        *,
        generation_enabled: bool,
        citation_validation_enabled: bool,
    ) -> GeneratedAnswer:
        if type(generation_enabled) is not bool:
            raise TypeError("generation_enabled must be boolean")
        if type(citation_validation_enabled) is not bool:
            raise TypeError("citation_validation_enabled must be boolean")
        started = self._clock()
        evidence = build_evidence(
            hits,
            top_k=self._config.evidence_top_k,
            per_item_chars=self._config.evidence_max_chars,
            total_chars=self._config.evidence_total_chars,
        )
        evidence_at = self._clock()
        if not generation_enabled:
            return GeneratedAnswer(
                status="disabled",
                answer=None,
                citations=(),
                evidence=evidence,
                validation_status=None,
                raw_content=None,
                request_id=None,
                model=None,
                usage=GenerationUsage(0, 0, 0, 0.0),
                timing_ms=GenerationTiming(
                    evidence=(evidence_at - started) * 1000.0,
                    generation=0.0,
                    validation=0.0,
                    end_to_end=(evidence_at - started) * 1000.0,
                ),
            )

        messages = build_messages(
            question,
            requested_version,
            [item.to_prompt_dict() for item in evidence],
        )
        completion = await self._get_client().generate(
            GenerationRequest(messages=messages, output_schema=output_schema())
        )
        generated_at = self._clock()
        try:
            parsed = parse_generation_output(completion.content)
            verified = verify_citations(
                parsed,
                evidence,
                semantic_validation_enabled=citation_validation_enabled,
            )
        except GenerationValidationError as error:
            failed_at = self._clock()
            raise PostResponseGenerationError(
                error,
                GenerationFailureContext(
                    raw_content=completion.content,
                    evidence=evidence,
                    request_id=completion.request_id,
                    model=completion.model,
                    usage=self._usage(
                        completion.input_tokens,
                        completion.cached_input_tokens,
                        completion.output_tokens,
                    ),
                    timing_ms=GenerationTiming(
                        evidence=(evidence_at - started) * 1000.0,
                        generation=(generated_at - evidence_at) * 1000.0,
                        validation=(failed_at - generated_at) * 1000.0,
                        end_to_end=(failed_at - started) * 1000.0,
                    ),
                    provider_fingerprint=completion.provider_fingerprint,
                ),
            ) from error
        validated_at = self._clock()
        return GeneratedAnswer(
            status="success",
            answer=verified.answer,
            citations=verified.citations,
            evidence=evidence,
            validation_status=verified.validation_status,
            raw_content=completion.content,
            request_id=completion.request_id,
            model=completion.model,
            usage=self._usage(
                completion.input_tokens,
                completion.cached_input_tokens,
                completion.output_tokens,
            ),
            provider_fingerprint=completion.provider_fingerprint,
            timing_ms=GenerationTiming(
                evidence=(evidence_at - started) * 1000.0,
                generation=(generated_at - evidence_at) * 1000.0,
                validation=(validated_at - generated_at) * 1000.0,
                end_to_end=(validated_at - started) * 1000.0,
            ),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
