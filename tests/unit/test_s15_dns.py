import pytest

from docguard.discovery.fetch import FetchError


def response(host, kind, values):
    return {
        "Status": 0,
        "Question": [{"name": host, "type": kind}],
        "Answer": [
            {"name": host, "type": kind, "TTL": 30, "data": value} for value in values
        ],
    }


def test_doh_resolves_both_families_and_caches_only_public_answers():
    from docguard.discovery.dns import DohResolver

    calls = []

    def query(host, kind):
        calls.append((host, kind))
        return response(
            host, kind, ["151.101.0.223"] if kind == 1 else ["2606:4700::1111"]
        )

    resolver = DohResolver(query=query)
    assert resolver("docs.python.org") == ["151.101.0.223", "2606:4700::1111"]
    assert resolver("docs.python.org") == resolver("docs.python.org")
    assert calls == [("docs.python.org", 1), ("docs.python.org", 28)]


@pytest.mark.parametrize("bad", ["127.0.0.1", "198.18.0.87", "10.0.0.1", "::1"])
def test_doh_rejects_non_public_answers_even_mixed_with_public(bad):
    from docguard.discovery.dns import DohResolver

    with pytest.raises(FetchError) as caught:
        DohResolver(query=lambda host, kind: response(host, kind, ["1.1.1.1", bad]))(
            "docs.python.org"
        )
    assert caught.value.reason_code == "dns_non_public"


def test_doh_does_not_fall_back_to_system_dns_on_failure():
    from docguard.discovery.dns import DohResolver

    def failed(host, kind):
        raise OSError("secret-provider-body")

    with pytest.raises(FetchError) as caught:
        DohResolver(query=failed)("docs.python.org")
    assert caught.value.reason_code == "dns_resolution_failed"
    assert "secret" not in str(caught.value)


def test_doh_rejects_mismatched_question_and_bad_status():
    from docguard.discovery.dns import DohResolver

    for value in (response("other.example.org", 1, ["1.1.1.1"]), {"Status": 2}):
        with pytest.raises(FetchError):
            DohResolver(query=lambda host, kind, value=value: value)("docs.python.org")


def test_dns_mode_is_explicit_and_validated(monkeypatch):
    from docguard.discovery.dns import DohResolver, configured_resolver
    from docguard.discovery.fetch import _resolve

    monkeypatch.delenv("DOCGUARD_DOCUMENT_DNS", raising=False)
    assert configured_resolver() is _resolve
    monkeypatch.setenv("DOCGUARD_DOCUMENT_DNS", "doh")
    assert isinstance(configured_resolver(), DohResolver)
    monkeypatch.setenv("DOCGUARD_DOCUMENT_DNS", "unsafe")
    with pytest.raises(ValueError):
        configured_resolver()


def test_auto_doh_uses_second_fixed_provider_after_transport_failure(monkeypatch):
    from docguard.discovery import dns

    calls = []

    def unavailable(host, kind):
        calls.append(("cloudflare", kind))
        raise TimeoutError("private network details")

    def available(host, kind):
        calls.append(("alidns", kind))
        payload = response(host, kind, ["1.1.1.1"] if kind == 1 else [])
        payload["Question"] = payload["Question"][0]
        return payload

    monkeypatch.setattr(dns, "_query", unavailable)
    monkeypatch.setattr(dns, "_query_alidns", available, raising=False)
    resolver = dns.DohResolver(provider="auto")
    assert resolver("pandas.pydata.org") == ["1.1.1.1"]
    assert resolver.last_provider == "alidns"
    assert calls == [("cloudflare", 1), ("alidns", 1), ("alidns", 28)]
    assert resolver("docs.python.org") == ["1.1.1.1"]
    assert calls[-2:] == [("alidns", 1), ("alidns", 28)]


def test_auto_doh_never_hides_private_answers_by_switching_provider(monkeypatch):
    from docguard.discovery import dns

    other_calls = []
    monkeypatch.setattr(
        dns, "_query", lambda host, kind: response(host, kind, ["198.18.0.1"])
    )
    monkeypatch.setattr(
        dns, "_query_alidns", lambda *args: other_calls.append(args), raising=False
    )
    with pytest.raises(FetchError) as caught:
        dns.DohResolver(provider="auto")("pandas.pydata.org")
    assert caught.value.reason_code == "dns_non_public"
    assert other_calls == []


def test_doh_provider_configuration_is_explicit_and_validated(monkeypatch):
    from docguard.discovery.dns import configured_resolver

    monkeypatch.setenv("DOCGUARD_DOCUMENT_DNS", "doh")
    monkeypatch.setenv("DOCGUARD_DOH_PROVIDER", "alidns")
    assert configured_resolver().provider == "alidns"
    monkeypatch.setenv("DOCGUARD_DOH_PROVIDER", "arbitrary-host")
    with pytest.raises(ValueError):
        configured_resolver()
