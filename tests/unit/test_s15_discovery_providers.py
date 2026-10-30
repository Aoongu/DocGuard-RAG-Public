import json
from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["type", "action"])
async def test_decision_normalizes_action_field_without_another_model_call(
    monkeypatch, field
):
    from types import SimpleNamespace

    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, request):
            calls.append(request)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        field: "search_documents",
                        "query": "pandas official documentation",
                    }
                ),
                model="deepseek-flash",
                input_tokens=997,
                cached_input_tokens=768,
                output_tokens=50,
                request_id="fixture",
                provider_fingerprint=None,
            )

        async def close(self):
            pass

    monkeypatch.setattr("docguard.discovery.providers.DeepSeekResponsesClient", Client)
    cfg = load_generation_config(
        Path(__file__).resolve().parents[2]
        / "configs/generation/s7-deepseek-flash-v1.json"
    )
    decision = await DeepSeekDecisionModel(cfg, "fixture").decide({})
    assert decision.action == {
        "action": "search_documents",
        "query": "pandas official documentation",
    }
    assert len(calls) == 1
    assert decision.input_tokens == 997


from docguard.discovery.providers import (
    BraveSearch,
    DeepSeekDecisionModel,
    ProviderError,
)
from docguard.generation.config import load_generation_config


@pytest.mark.asyncio
async def test_brave_request_filters_private_urls_and_deduplicates():
    def handler(request):
        assert request.url.host == "api.search.brave.com"
        assert request.headers["X-Subscription-Token"] == "fixture-key"
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {"title": "Docs", "url": "https://docs.python.org/3/"},
                        {"title": "private", "url": "https://127.0.0.1/"},
                        {"title": "duplicate", "url": "https://docs.python.org/3/"},
                    ]
                }
            },
        )

    result = await BraveSearch(
        "fixture-key", transport=httpx.MockTransport(handler)
    ).search("Python", 5)
    assert len(result) == 1 and result[0].url == "https://docs.python.org/3/"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_brave_failure_is_sanitized_without_retry(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="provider echoed fixture-key secret")

    with pytest.raises(ProviderError) as captured:
        await BraveSearch("fixture-key", transport=httpx.MockTransport(handler)).search(
            "Python"
        )
    assert len(calls) == 1
    assert "fixture-key" not in str(captured.value)


@pytest.mark.asyncio
async def test_decision_schema_restricts_actions_to_server_available_tools():
    captured = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, request):
            captured.append(request.output_schema["properties"]["action"]["enum"])
            raise RuntimeError("Stop after inspecting request schema")

        async def close(self):
            pass

    from unittest.mock import patch

    config = load_generation_config(
        Path(__file__).resolve().parents[2]
        / "configs/generation/s7-deepseek-flash-v1.json"
    )
    with (
        patch("docguard.discovery.providers.DeepSeekResponsesClient", Client),
        pytest.raises(ProviderError),
    ):
        await DeepSeekDecisionModel(config, "fixture").decide(
            {"available_actions": ["search_documents", "ask_user"]}
        )
    assert captured == [["search_documents", "ask_user"]]


@pytest.mark.asyncio
async def test_deepseek_decision_uses_action_schema_and_reports_usage():
    cfg = load_generation_config(
        Path(__file__).resolve().parents[2]
        / "configs/generation/s7-deepseek-flash-v1.json"
    )

    def handler(request):
        payload = json.loads(request.content)
        assert "search_documents" in json.dumps(payload)
        return httpx.Response(
            200,
            json={
                "id": "decision-fixture",
                "object": "response",
                "status": "completed",
                "model": "deepseek-flash",
                "output": [
                    {
                        "type": "message",
                        "id": "msg",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"action":"ask_user","question":"Which version?"}',
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
            },
        )

    decision = await DeepSeekDecisionModel(
        cfg, "fixture-key", transport=httpx.MockTransport(handler)
    ).decide({"goal": "Find docs"})
    assert decision.action["action"] == "ask_user"
    assert decision.input_tokens == 120 and decision.cached_input_tokens == 20
    assert decision.cost_usd > 0
