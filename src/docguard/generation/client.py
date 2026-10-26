"""OpenAI-compatible strict structured-generation client for S7."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from docguard.generation.config import GenerationConfig, normalize_base_url
from docguard.generation.models import GenerationCompletion, GenerationRequest


class GenerationClientError(RuntimeError):
    """Base error for sanitized generation-client failures."""


class GenerationAuthenticationError(GenerationClientError):
    """Authentication or authorization failed."""


class GenerationRateLimitError(GenerationClientError):
    """The provider rejected the request because of rate limits."""


class GenerationUpstreamError(GenerationClientError):
    """The provider or transport failed."""


class GenerationResponseError(GenerationClientError):
    """The provider returned an invalid completion payload."""


class GenerationClientClosedError(GenerationClientError):
    """Generation was attempted after client closure."""


class StructuredGenerationClient(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationCompletion: ...

    async def close(self) -> None: ...


def _non_negative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise GenerationResponseError(f"generation usage {field} must be non-negative")
    return value


def _parse_completion(payload: object, expected_model: str) -> GenerationCompletion:
    if not isinstance(payload, dict) or not all(
        field in payload for field in ("id", "model", "choices", "usage")
    ):
        raise GenerationResponseError("generation response fields are incomplete")
    request_id = payload["id"]
    model = payload["model"]
    choices = payload["choices"]
    usage = payload["usage"]
    if not isinstance(request_id, str) or not request_id:
        raise GenerationResponseError("generation request id is invalid")
    if model != expected_model:
        raise GenerationResponseError("generation response model differs from config")
    if not isinstance(choices, list) or len(choices) != 1:
        raise GenerationResponseError(
            "generation response must contain exactly one choice"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise GenerationResponseError("generation choice must be an object")
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise GenerationResponseError("generation finish_reason must be stop")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise GenerationResponseError("generation message must be an object")
    if message.get("refusal") is not None:
        raise GenerationResponseError("generation provider returned a refusal")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise GenerationResponseError("generation content must be non-empty text")
    if not isinstance(usage, dict):
        raise GenerationResponseError("generation usage must be an object")
    input_tokens = _non_negative_int(usage.get("prompt_tokens"), "prompt_tokens")
    output_tokens = _non_negative_int(
        usage.get("completion_tokens"), "completion_tokens"
    )
    details = usage.get("prompt_tokens_details", {})
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise GenerationResponseError(
            "generation prompt token details must be an object"
        )
    cached_tokens = _non_negative_int(details.get("cached_tokens", 0), "cached_tokens")
    if cached_tokens > input_tokens:
        raise GenerationResponseError("cached input tokens cannot exceed input tokens")
    return GenerationCompletion(
        content=content,
        request_id=request_id,
        model=model,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
    )


def _parse_deepseek_response(
    payload: object, expected_model: str
) -> GenerationCompletion:
    if not isinstance(payload, dict) or not all(
        field in payload for field in ("id", "status", "model", "output", "usage")
    ):
        raise GenerationResponseError("generation response fields are incomplete")
    request_id = payload["id"]
    if not isinstance(request_id, str) or not request_id:
        raise GenerationResponseError("generation request id is invalid")
    if payload["status"] != "completed":
        raise GenerationResponseError("generation response status must be completed")
    if payload["model"] != expected_model:
        raise GenerationResponseError("generation response model differs from config")
    output = payload["output"]
    if not isinstance(output, list):
        raise GenerationResponseError("generation output must be an array")
    messages = [
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "message"
    ]
    if len(messages) != 1:
        raise GenerationResponseError(
            "generation response must contain one assistant message"
        )
    message = messages[0]
    if (
        message.get("status") != "completed"
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), list)
    ):
        raise GenerationResponseError("generation assistant message is invalid")
    content_parts = message["content"]
    output_texts = [
        part.get("text")
        for part in content_parts
        if isinstance(part, dict) and part.get("type") == "output_text"
    ]
    if (
        len(output_texts) != 1
        or not isinstance(output_texts[0], str)
        or not output_texts[0].strip()
    ):
        raise GenerationResponseError(
            "generation response must contain one non-empty output_text"
        )
    usage = payload["usage"]
    if not isinstance(usage, dict):
        raise GenerationResponseError("generation usage must be an object")
    input_tokens = _non_negative_int(usage.get("input_tokens"), "input_tokens")
    output_tokens = _non_negative_int(usage.get("output_tokens"), "output_tokens")
    details = usage.get("input_tokens_details", {})
    if not isinstance(details, dict):
        raise GenerationResponseError(
            "generation input token details must be an object"
        )
    cached_tokens = _non_negative_int(details.get("cached_tokens", 0), "cached_tokens")
    if cached_tokens > input_tokens:
        raise GenerationResponseError("cached input tokens cannot exceed input tokens")
    fingerprint = payload.get("system_fingerprint")
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or not fingerprint.strip()
    ):
        raise GenerationResponseError("generation provider fingerprint is invalid")
    return GenerationCompletion(
        content=output_texts[0],
        request_id=request_id,
        model=payload["model"],
        finish_reason="completed",
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        provider_fingerprint=fingerprint,
    )


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code in {401, 403}:
        raise GenerationAuthenticationError(
            f"generation authentication failed with HTTP {response.status_code}"
        )
    if response.status_code == 429:
        raise GenerationRateLimitError("generation rate limited with HTTP 429")
    if response.status_code < 200 or response.status_code >= 300:
        raise GenerationUpstreamError(
            f"generation provider failed with HTTP {response.status_code}"
        )


class OpenAICompatibleClient:
    """One-shot Chat Completions adapter with no implicit retries."""

    def __init__(
        self,
        config: GenerationConfig,
        *,
        api_key: str,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("API key must be a non-empty string")
        self._config = config
        self._closed = False
        root = normalize_base_url(base_url or config.default_base_url)
        self._client = httpx.AsyncClient(
            base_url=root + "/",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=httpx.Timeout(config.timeout_seconds),
            transport=transport,
        )

    async def generate(self, request: GenerationRequest) -> GenerationCompletion:
        if self._closed:
            raise GenerationClientClosedError("generation client is closed")
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": [dict(message) for message in request.messages],
            "temperature": self._config.temperature,
            "max_completion_tokens": self._config.max_output_tokens,
            "reasoning_effort": self._config.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "docguard_answer",
                    "strict": True,
                    "schema": request.output_schema,
                },
            },
        }
        try:
            response = await self._client.post("chat/completions", json=body)
        except httpx.HTTPError as error:
            raise GenerationUpstreamError("generation transport failed") from error
        _raise_for_status(response)
        try:
            payload = response.json()
        except ValueError as error:
            raise GenerationResponseError(
                "generation response is not valid JSON"
            ) from error
        return _parse_completion(payload, self._config.model)

    async def close(self) -> None:
        if not self._closed:
            await self._client.aclose()
            self._closed = True


class DeepSeekResponsesClient:
    """One-shot DeepSeek Responses adapter with no implicit retries."""

    def __init__(
        self,
        config: GenerationConfig,
        *,
        api_key: str,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if config.protocol != "deepseek-responses":
            raise ValueError("DeepSeek client requires a deepseek-responses config")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("API key must be a non-empty string")
        self._config = config
        self._closed = False
        root = normalize_base_url(base_url or config.default_base_url)
        self._client = httpx.AsyncClient(
            base_url=root + "/",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=httpx.Timeout(config.timeout_seconds),
            transport=transport,
        )

    async def generate(self, request: GenerationRequest) -> GenerationCompletion:
        if self._closed:
            raise GenerationClientClosedError("generation client is closed")
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        body: dict[str, Any] = {
            "model": self._config.model,
            "input": [dict(message) for message in request.messages],
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
            "reasoning": {"effort": self._config.reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "docguard_answer",
                    "schema": request.output_schema,
                }
            },
        }
        try:
            response = await self._client.post("responses", json=body)
        except httpx.HTTPError as error:
            raise GenerationUpstreamError("generation transport failed") from error
        _raise_for_status(response)
        try:
            payload = response.json()
        except ValueError as error:
            raise GenerationResponseError(
                "generation response is not valid JSON"
            ) from error
        return _parse_deepseek_response(payload, self._config.model)

    async def close(self) -> None:
        if not self._closed:
            await self._client.aclose()
            self._closed = True


class FixedGenerationClient:
    """Deterministic in-memory client used by offline tests and smoke runs."""

    def __init__(self, completions: Sequence[GenerationCompletion]) -> None:
        self._completions = tuple(completions)
        self._index = 0
        self._closed = False

    async def generate(self, request: GenerationRequest) -> GenerationCompletion:
        if self._closed:
            raise GenerationClientClosedError("generation client is closed")
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        if self._index >= len(self._completions):
            raise GenerationResponseError("fixed generation client is exhausted")
        completion = self._completions[self._index]
        self._index += 1
        return completion

    async def close(self) -> None:
        self._closed = True
