import json
from pathlib import Path

import httpx
import pytest

from docguard.generation.client import (
    DeepSeekResponsesClient,
    FixedGenerationClient,
    GenerationAuthenticationError,
    GenerationClientClosedError,
    GenerationRateLimitError,
    GenerationResponseError,
    GenerationUpstreamError,
    OpenAICompatibleClient,
)
from docguard.generation.config import load_generation_config
from docguard.generation.models import GenerationCompletion, GenerationRequest
from docguard.generation.prompt import output_schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_generation_config(
    PROJECT_ROOT / "configs" / "generation" / "s7-generation-v1.json"
)
DEEPSEEK_CONFIG = load_generation_config(
    PROJECT_ROOT / "configs" / "generation" / "s7-deepseek-flash-v1.json"
)
SECRET = "sk-test-do-not-log"


def _request() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Question and evidence."},
        ),
        output_schema=output_schema(),
    )


def _response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "chatcmpl-fixed",
        "model": CONFIG.model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"WAL detail.","citations":["E01"]}',
                    "refusal": None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
    }
    payload.update(overrides)
    return payload


def _deepseek_response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "resp-fixed",
        "object": "response",
        "status": "completed",
        "model": "deepseek-flash",
        "system_fingerprint": "fp_test_1017",
        "output": [
            {
                "type": "message",
                "id": "msg-fixed",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"answer":"WAL detail.","citations":["E01"]}',
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 120,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens": 18,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 138,
        },
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_deepseek_responses_client_sends_exact_request_and_parses_identity() -> (
    None
):
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_deepseek_response())

    client = DeepSeekResponsesClient(
        DEEPSEEK_CONFIG,
        api_key=SECRET,
        transport=httpx.MockTransport(handler),
    )
    completion = await client.generate(_request())
    await client.close()

    assert captured["url"] == "https://api.deepseek.com/responses"
    assert captured["authorization"] == f"Bearer {SECRET}"
    assert captured["json"] == {
        "model": "deepseek-flash",
        "input": list(_request().messages),
        "temperature": 0.0,
        "max_output_tokens": 800,
        "reasoning": {"effort": "none"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "docguard_answer",
                "schema": output_schema(),
            }
        },
    }
    assert completion == GenerationCompletion(
        content='{"answer":"WAL detail.","citations":["E01"]}',
        request_id="resp-fixed",
        model="deepseek-flash",
        finish_reason="completed",
        input_tokens=120,
        cached_input_tokens=20,
        output_tokens=18,
        provider_fingerprint="fp_test_1017",
    )


@pytest.mark.asyncio
async def test_deepseek_responses_client_allows_absent_fingerprint() -> None:
    payload = _deepseek_response()
    payload.pop("system_fingerprint")

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = DeepSeekResponsesClient(
        DEEPSEEK_CONFIG, api_key=SECRET, transport=httpx.MockTransport(handler)
    )
    completion = await client.generate(_request())
    await client.close()
    assert completion.provider_fingerprint is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"not": "a response"}, "response fields"),
        (_deepseek_response(status="incomplete"), "status"),
        (_deepseek_response(model="DeepSeek-V4.1-Flash-1017"), "model"),
        (_deepseek_response(output=[]), "assistant message"),
        (
            _deepseek_response(output=_deepseek_response()["output"] * 2),
            "assistant message",
        ),
        (_deepseek_response(usage=None), "usage"),
        (_deepseek_response(system_fingerprint=123), "fingerprint"),
    ],
)
async def test_deepseek_responses_client_rejects_invalid_payloads(
    payload: dict[str, object], message: str
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = DeepSeekResponsesClient(
        DEEPSEEK_CONFIG, api_key=SECRET, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(GenerationResponseError, match=message):
        await client.generate(_request())
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, GenerationAuthenticationError),
        (403, GenerationAuthenticationError),
        (429, GenerationRateLimitError),
        (500, GenerationUpstreamError),
    ],
)
async def test_deepseek_responses_client_sanitizes_http_errors(
    status: int, error_type: type[Exception]
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=f"provider echoed {SECRET}")

    client = DeepSeekResponsesClient(
        DEEPSEEK_CONFIG, api_key=SECRET, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(error_type) as captured:
        await client.generate(_request())
    await client.close()
    assert SECRET not in str(captured.value)
    assert "provider echoed" not in str(captured.value)


@pytest.mark.asyncio
async def test_openai_client_sends_exact_structured_request_and_parses_usage() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_response())

    client = OpenAICompatibleClient(
        CONFIG,
        api_key=SECRET,
        transport=httpx.MockTransport(handler),
    )
    completion = await client.generate(_request())
    await client.close()

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["authorization"] == f"Bearer {SECRET}"
    body = captured["json"]
    assert isinstance(body, dict)
    assert body == {
        "model": CONFIG.model,
        "messages": list(_request().messages),
        "temperature": 0.0,
        "max_completion_tokens": 800,
        "reasoning_effort": "none",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "docguard_answer",
                "strict": True,
                "schema": output_schema(),
            },
        },
    }
    assert completion == GenerationCompletion(
        content='{"answer":"WAL detail.","citations":["E01"]}',
        request_id="chatcmpl-fixed",
        model=CONFIG.model,
        finish_reason="stop",
        input_tokens=120,
        cached_input_tokens=20,
        output_tokens=18,
    )


def test_openai_client_requires_non_empty_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleClient(CONFIG, api_key="  ")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, GenerationAuthenticationError),
        (403, GenerationAuthenticationError),
        (429, GenerationRateLimitError),
        (500, GenerationUpstreamError),
    ],
)
async def test_openai_client_classifies_http_errors_without_leaking_body_or_key(
    status: int, error_type: type[Exception]
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=f"provider echoed {SECRET}")

    client = OpenAICompatibleClient(
        CONFIG, api_key=SECRET, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(error_type) as captured:
        await client.generate(_request())
    await client.close()

    assert SECRET not in str(captured.value)
    assert "provider echoed" not in str(captured.value)


@pytest.mark.asyncio
async def test_openai_client_maps_transport_failures_without_key_leak() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect with {SECRET}", request=request)

    client = OpenAICompatibleClient(
        CONFIG, api_key=SECRET, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(GenerationUpstreamError) as captured:
        await client.generate(_request())
    await client.close()
    assert SECRET not in str(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"not": "a completion"}, "response fields"),
        (_response(choices=[]), "exactly one choice"),
        (_response(choices=[_response()["choices"][0]] * 2), "exactly one choice"),
        (_response(model="rolling-alias"), "model"),
        (_response(usage=None), "usage"),
        (
            _response(
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "{}",
                            "refusal": None,
                        },
                    }
                ]
            ),
            "finish_reason",
        ),
        (
            _response(
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "{}",
                            "refusal": "blocked",
                        },
                    }
                ]
            ),
            "refusal",
        ),
    ],
)
async def test_openai_client_rejects_malformed_or_incomplete_responses(
    payload: dict[str, object], message: str
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = OpenAICompatibleClient(
        CONFIG, api_key=SECRET, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(GenerationResponseError, match=message):
        await client.generate(_request())
    await client.close()


@pytest.mark.asyncio
async def test_openai_client_rejects_non_json_response() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    client = OpenAICompatibleClient(
        CONFIG, api_key=SECRET, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(GenerationResponseError, match="valid JSON"):
        await client.generate(_request())
    await client.close()


@pytest.mark.asyncio
async def test_openai_client_cannot_generate_after_close() -> None:
    client = OpenAICompatibleClient(CONFIG, api_key=SECRET)
    await client.close()
    await client.close()
    with pytest.raises(GenerationClientClosedError):
        await client.generate(_request())


@pytest.mark.asyncio
async def test_fixed_generation_client_returns_values_in_order_and_closes() -> None:
    completions = (
        GenerationCompletion("one", "r1", CONFIG.model, "stop", 1, 0, 2),
        GenerationCompletion("two", "r2", CONFIG.model, "stop", 3, 1, 4),
    )
    client = FixedGenerationClient(completions)
    assert await client.generate(_request()) == completions[0]
    assert await client.generate(_request()) == completions[1]
    with pytest.raises(GenerationResponseError, match="exhausted"):
        await client.generate(_request())
    await client.close()
    with pytest.raises(GenerationClientClosedError):
        await client.generate(_request())
