import logging
from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
async def test_serpapi_filters_results_and_redacts_key_from_http_logs(caplog):
    from docguard.discovery.providers import SerpAPISearch

    caplog.set_level(logging.INFO, logger="httpx")

    def handler(request):
        assert request.url.host == "serpapi.com"
        assert request.url.path == "/search.json"
        assert request.url.params["engine"] == "google"
        assert request.url.params["api_key"] == "fixture-serp-secret"
        assert request.url.params["q"] == "Python configparser official docs"
        return httpx.Response(
            200,
            json={
                "search_metadata": {"status": "Success"},
                "organic_results": [
                    {
                        "title": "Python",
                        "link": "https://docs.python.org/3/",
                        "snippet": "Official docs",
                    },
                    {"link": "https://127.0.0.1/"},
                    {"link": "https://docs.python.org/3/"},
                ],
            },
        )

    result = await SerpAPISearch(
        "fixture-serp-secret", transport=httpx.MockTransport(handler)
    ).search("Python configparser official docs", 5)
    assert len(result) == 1 and result[0].url == "https://docs.python.org/3/"
    assert "fixture-serp-secret" not in caplog.text
    assert "api_key=" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,payload",
    [
        (401, {}),
        (429, {}),
        (500, {}),
        (200, {"error": "fixture-serp-secret"}),
        (200, {"organic_results": "broken"}),
    ],
)
async def test_serpapi_failures_are_sanitized_without_retry(status, payload):
    from docguard.discovery.providers import ProviderError, SerpAPISearch

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=payload)

    with pytest.raises(ProviderError) as caught:
        await SerpAPISearch(
            "fixture-serp-secret", transport=httpx.MockTransport(handler)
        ).search("Python")
    assert len(calls) == 1
    assert "fixture-serp-secret" not in str(caught.value)
    expected = {
        401: "search_authentication_failed",
        429: "search_rate_or_quota_limited",
        500: "search_upstream_failed",
    }.get(status, "search_invalid_response")
    assert str(caught.value) == expected


@pytest.mark.asyncio
async def test_serpapi_network_error_is_safe_and_distinct():
    from docguard.discovery.providers import ProviderError, SerpAPISearch

    def handler(request):
        raise httpx.ConnectError("fixture-serp-secret", request=request)

    with pytest.raises(ProviderError, match="^search_network_failed$"):
        await SerpAPISearch(
            "fixture-serp-secret", transport=httpx.MockTransport(handler)
        ).search("Java")


def test_runtime_selects_serpapi_explicitly_and_never_reuses_deepseek_key(
    tmp_path, monkeypatch
):
    from docguard.discovery.providers import SerpAPISearch
    from docguard.discovery.runtime import build_discovery_service

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("DOCGUARD_SEARCH_PROVIDER", "serpapi")
    monkeypatch.setenv("SERPAPI_API_KEY", "fixture-serp-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "different-model-secret")
    service = build_discovery_service(root, tmp_path / "first")
    assert isinstance(service.agent.search, SerpAPISearch)
    assert service.capabilities()["search_provider"] == "serpapi"
    assert service.capabilities()["agent_enabled"]
    monkeypatch.delenv("SERPAPI_API_KEY")
    unavailable = build_discovery_service(root, tmp_path / "second")
    assert not unavailable.capabilities()["search_configured"]
    assert unavailable.capabilities()["decision_model_configured"]


def test_serpapi_is_the_project_default_when_no_override_is_set(tmp_path, monkeypatch):
    from docguard.discovery.runtime import build_discovery_service

    monkeypatch.delenv("DOCGUARD_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "fixture-search-secret")
    service = build_discovery_service(Path(__file__).resolve().parents[2], tmp_path)
    assert service.capabilities()["search_provider"] == "serpapi"
    assert service.capabilities()["search_configured"]


@pytest.mark.asyncio
async def test_switching_search_provider_cannot_resume_an_old_agent_checkpoint(
    tmp_path,
):
    from docguard.discovery.importer import DiscoveryService
    from docguard.discovery.providers import Decision

    class Model:
        async def decide(self, context):
            return Decision(
                {"action": "ask_user", "question": "Which version?"}, "mock"
            )

    class Search:
        provider_id = "first-provider"

        async def search(self, query, limit=5):
            return []

    search = Search()
    service = DiscoveryService(
        tmp_path, lambda: None, decision_model=Model(), search_provider=search
    )
    row = service.store.create(
        {
            "goal": "Find Python docs",
            "product": "Python",
            "version": None,
            "urls": [],
            "mode": "agent",
        }
    )
    await service.agent.run(row["task_id"])
    assert service.get_task(row["task_id"])["status"] == "awaiting_input"
    search.provider_id = "second-provider"
    await service.agent.run(row["task_id"])
    final = service.get_task(row["task_id"])
    assert (
        final["status"] == "failed"
        and final["stop_reason"] == "agent_configuration_changed"
    )
