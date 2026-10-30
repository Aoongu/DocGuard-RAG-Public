import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def explicit_legacy_search_provider(monkeypatch):
    monkeypatch.setenv("DOCGUARD_SEARCH_PROVIDER", "brave")


def test_preflight_uses_the_same_opt_in_doh_resolver_as_runtime(monkeypatch):
    from docguard.discovery import dns
    from docguard.discovery.preflight import inspect_readiness

    monkeypatch.setenv("DOCGUARD_DOCUMENT_DNS", "doh")
    calls = []

    def query(host, kind):
        calls.append((host, kind))
        return {
            "Status": 0,
            "Question": [{"name": host, "type": kind}],
            "Answer": [{"name": host, "type": kind, "TTL": 30, "data": "1.1.1.1"}],
        }

    monkeypatch.setattr(dns, "_query", query)
    report = inspect_readiness(ROOT, check_dns=True)
    assert report["document_dns_mode"] == "doh"
    assert report["dns_status"] == "passed"
    assert len(calls) == 8


def test_offline_probe_uses_process_keys_without_network_or_project_writes(monkeypatch):
    from docguard.discovery import preflight
    from docguard.discovery.preflight import inspect_readiness

    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-test-model-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "private-test-search-key")

    def forbidden(host):
        raise AssertionError("DNS must be opt-in")

    original_builder = preflight.build_discovery_service

    def isolated_builder(root, storage):
        assert not storage.resolve().is_relative_to(ROOT)
        return original_builder(root, storage)

    monkeypatch.setattr(preflight, "build_discovery_service", isolated_builder)
    report = inspect_readiness(ROOT, resolver=forbidden)
    assert report["local_import_configured"]
    assert report["capabilities"]["agent_enabled"]
    assert report["agent_preflight_ready"] is None
    assert report["dns_status"] == "not_checked"
    assert report["external_api_calls"] == 0
    assert "private-test" not in json.dumps(report)


def test_dns_rejects_mixed_public_and_fake_ip_without_fetching(monkeypatch):
    from docguard.discovery.preflight import inspect_readiness

    monkeypatch.setenv("DEEPSEEK_API_KEY", "model-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "search-key")
    calls = []

    def resolve(host):
        calls.append(host)
        return ["1.1.1.1", "198.18.0.2"] if host == "docs.python.org" else ["1.1.1.1"]

    report = inspect_readiness(ROOT, check_dns=True, resolver=resolve)
    assert sorted(calls) == [
        "docs.oracle.com",
        "docs.python.org",
        "redis.io",
        "www.postgresql.org",
    ]
    assert report["dns"]["docs.python.org"]["status"] == "blocked_non_public"
    assert not report["agent_preflight_ready"]
    assert report["provider_authentication"] == "not_checked"
    assert report["https_robots_html"] == "not_checked"
    public = inspect_readiness(ROOT, check_dns=True, resolver=lambda _: ["1.1.1.1"])
    assert public["agent_preflight_ready"]

    def unavailable(host):
        raise OSError("private-secret-provider-message")

    failed = inspect_readiness(ROOT, check_dns=True, resolver=unavailable)
    assert not failed["agent_preflight_ready"]
    assert "private-secret" not in json.dumps(failed)


def test_failures_are_sanitized_and_openai_key_does_not_enable_discovery(
    tmp_path, monkeypatch
):
    from docguard.discovery.preflight import inspect_readiness

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    report = inspect_readiness(ROOT)
    assert not report["capabilities"]["decision_model_configured"]
    assert not report["agent_preflight_ready"]
    missing = inspect_readiness(tmp_path)
    serialized = json.dumps(missing)
    assert not missing["local_import_configured"]
    assert str(tmp_path) not in serialized
    assert "must-not-be-used" not in serialized
    assert missing["checks"]["tokenizers"]["status"] == "failed"


def test_cli_requires_complete_agent_preflight_and_reports_missing_assets(
    tmp_path, monkeypatch, capsys
):
    from docguard.discovery.preflight import main

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert main(["--project-root", str(ROOT), "--require-agent"]) == 2
    assert json.loads(capsys.readouterr().out)["local_import_configured"]
    assert main(["--project-root", str(tmp_path)]) == 2
    assert not json.loads(capsys.readouterr().out)["local_import_configured"]
