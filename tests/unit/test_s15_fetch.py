import pytest

from docguard.discovery.fetch import (
    FetchError,
    FetchResponse,
    SafeFetcher,
    validate_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a",
        "https://user:pass@example.com/a",
        "https://localhost/a",
        "https://127.0.0.1/a",
        "https://[::1]/a",
        "https://example.com:8443/a",
        "file:///x",
        "https://example.com/a\nHost:x",
    ],
)
def test_url_rejects_unsupported_or_local_targets(url):
    with pytest.raises(FetchError):
        validate_url(url)


def test_fetch_preserves_query_and_removes_fragment():
    calls = []

    def transport(url, address, limit, timeout):
        calls.append((url, address))
        if url.endswith("/robots.txt"):
            return FetchResponse(404, {}, b"")
        return FetchResponse(
            200, {"content-type": "text/html"}, b"<main><p>text</p></main>"
        )

    result = SafeFetcher(
        resolver=lambda host: ["93.184.216.34"], transport=transport, interval=0
    ).fetch("https://example.com/a?v=1#part")
    assert result.final_url == "https://example.com/a?v=1"
    assert calls[-1] == ("https://example.com/a?v=1", "93.184.216.34")


def test_dns_private_resolution_is_rejected_before_transport():
    def forbidden(*args):
        pytest.fail("transport must not be called")

    with pytest.raises(FetchError, match="address") as caught:
        SafeFetcher(resolver=lambda host: ["10.0.0.1"], transport=forbidden).fetch(
            "https://example.com/"
        )
    assert caught.value.reason_code == "dns_non_public"


def test_redirect_to_private_or_other_site_is_rejected():
    def transport(url, address, limit, timeout):
        if url.endswith("/robots.txt"):
            return FetchResponse(404, {}, b"")
        return FetchResponse(302, {"location": "https://127.0.0.1/admin"}, b"")

    with pytest.raises(FetchError):
        SafeFetcher(
            resolver=lambda host: ["93.184.216.34"], transport=transport, interval=0
        ).fetch("https://example.com/docs")


@pytest.mark.parametrize(
    "response",
    [
        FetchResponse(200, {"content-type": "application/pdf"}, b"pdf"),
        FetchResponse(200, {"content-type": "text/html"}, b"x" * 129),
        FetchResponse(403, {}, b""),
        FetchResponse(
            200, {"content-type": "text/html", "content-encoding": "gzip"}, b"zip"
        ),
    ],
)
def test_fetch_rejects_bad_response(response):
    def transport(url, address, limit, timeout):
        return FetchResponse(404, {}, b"") if url.endswith("/robots.txt") else response

    with pytest.raises(FetchError):
        SafeFetcher(
            resolver=lambda host: ["93.184.216.34"],
            transport=transport,
            max_bytes=128,
            interval=0,
        ).fetch("https://example.com/docs")


def test_robots_denial_blocks_document_request():
    calls = []

    def transport(url, address, limit, timeout):
        calls.append(url)
        return FetchResponse(200, {}, b"User-agent: *\nDisallow: /private")

    with pytest.raises(FetchError, match="robots"):
        SafeFetcher(
            resolver=lambda host: ["93.184.216.34"], transport=transport, interval=0
        ).fetch("https://example.com/private")
    assert calls == ["https://example.com/robots.txt"]


def test_fetch_uses_alternate_public_ip_and_remembers_success():
    calls = []

    def transport(url, address, limit, timeout):
        calls.append((url, address, timeout))
        if address == "93.184.216.34":
            raise TimeoutError("fixture timeout")
        return (
            FetchResponse(404, {}, b"")
            if url.endswith("robots.txt")
            else FetchResponse(200, {"content-type": "text/html"}, b"<main>docs</main>")
        )

    result = SafeFetcher(
        resolver=lambda host: ["93.184.216.34", "8.8.8.8"],
        transport=transport,
        interval=0,
    ).fetch("https://example.com/guide")
    assert result.body == b"<main>docs</main>"
    assert [c[1] for c in calls] == ["93.184.216.34", "8.8.8.8", "8.8.8.8"]
    assert all(0 < c[2] <= 5 for c in calls)


def test_fetch_network_failure_is_bounded_and_has_safe_diagnostics():
    calls = []

    def transport(url, address, limit, timeout):
        calls.append(address)
        raise TimeoutError("fixture must not leak")

    with pytest.raises(FetchError) as caught:
        SafeFetcher(
            resolver=lambda host: ["8.8.8.8"], transport=transport, interval=0
        ).fetch("https://example.com/guide")
    assert calls == ["8.8.8.8"] * 3
    assert len(caught.value.attempts) == 3
    assert caught.value.attempts[0]["error_type"] == "TimeoutError"
    assert "fixture must not leak" not in str(caught.value.attempts)


def test_mixed_public_and_private_dns_does_not_try_even_public_ip():
    with pytest.raises(FetchError) as caught:
        SafeFetcher(
            resolver=lambda host: ["8.8.8.8", "127.0.0.1"],
            transport=lambda *args: pytest.fail("unsafe DNS set"),
        ).fetch("https://example.com/")
    assert caught.value.reason_code == "dns_non_public"


def test_network_attempts_share_one_request_deadline(monkeypatch):
    import docguard.discovery.fetch as module

    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    calls = []

    def transport(url, address, limit, timeout):
        calls.append(timeout)
        now[0] += timeout
        raise TimeoutError("timeout")

    with pytest.raises(FetchError):
        SafeFetcher(
            resolver=lambda host: ["8.8.8.8", "1.1.1.1"],
            transport=transport,
            interval=0,
            timeout=4,
        ).fetch("https://example.com/")
    assert calls == [4]


def test_certificate_verification_failure_is_not_retried():
    import ssl

    calls = []

    def transport(url, address, limit, timeout):
        calls.append(address)
        raise ssl.SSLCertVerificationError(1, "fixture certificate error")

    with pytest.raises(FetchError) as caught:
        SafeFetcher(
            resolver=lambda host: ["8.8.8.8", "1.1.1.1"],
            transport=transport,
            interval=0,
        ).fetch("https://example.com/")
    assert calls == ["8.8.8.8"]
    assert caught.value.attempts[0]["error_type"] == "SSLCertVerificationError"
