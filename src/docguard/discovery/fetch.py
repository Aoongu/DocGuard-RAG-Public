"""Bounded HTTPS fetcher with pinned public IP connections and robots checks."""

import hashlib
import http.client
import ipaddress
import socket
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser


class FetchError(ValueError):
    """A source cannot be safely or reliably downloaded."""

    def __init__(
        self, message, *, reason_code="source_inspection_failed", attempts=None
    ):
        super().__init__(message)
        self.attempts = attempts or []
        self.reason_code = (
            reason_code
            if reason_code
            in {
                "dns_non_public",
                "dns_resolution_failed",
                "https_download_failed",
                "robots_unavailable",
                "robots_disallowed",
                "not_html",
            }
            else "source_inspection_failed"
        )


def validate_url(url: str) -> str:
    if (
        not isinstance(url, str)
        or any(ord(c) <= 32 or ord(c) == 127 for c in url)
        or "\\" in url
    ):
        raise FetchError("Invalid URL")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        if (
            parts.scheme != "https"
            or not host
            or parts.username is not None
            or parts.password is not None
            or parts.port not in (None, 443)
        ):
            raise FetchError("Only public HTTPS URLs on port 443 are supported")
        host = host.encode("idna").decode("ascii").lower().rstrip(".")
        if (
            host == "localhost"
            or host.endswith((".localhost", ".local", ".internal"))
            or "%" in host
        ):
            raise FetchError("Local hosts are not allowed")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise FetchError("Non-public address")
        authority = f"[{host}]" if ":" in host else host
        return urlunsplit(("https", authority, parts.path or "/", parts.query, ""))
    except (ValueError, UnicodeError) as error:
        raise FetchError("Invalid or unsupported URL") from error


@dataclass(frozen=True, slots=True)
class FetchResponse:
    status: int
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class Snapshot:
    requested_url: str
    final_url: str
    retrieved_at: str
    content_sha256: str
    body: bytes
    robots_url: str
    robots_status: int
    robots_sha256: str


def _resolve(host: str) -> list[str]:
    return sorted(
        {row[4][0] for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    )


def _transport(url: str, address: str, limit: int, timeout: float) -> FetchResponse:
    parts = urlsplit(url)
    context = ssl.create_default_context()
    connection = http.client.HTTPSConnection(
        parts.hostname, timeout=timeout, context=context
    )
    deadline = time.monotonic() + timeout
    raw = None
    phase = "tcp_connect"

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("HTTPS attempt deadline exceeded")
        return value

    try:
        raw = socket.create_connection((address, 443), timeout=remaining())
        raw.settimeout(remaining())
        phase = "tls_handshake"
        connection.sock = context.wrap_socket(raw, server_hostname=parts.hostname)
        connection.sock.settimeout(remaining())
        phase = "http_request"
        connection.request(
            "GET",
            urlunsplit(("", "", parts.path or "/", parts.query, "")),
            headers={
                "User-Agent": "DocGuard/0.1",
                "Accept": "text/html,text/plain",
                "Accept-Encoding": "identity",
            },
        )
        phase = "http_headers"
        connection.sock.settimeout(remaining())
        response = connection.getresponse()
        phase = "http_body"
        body = bytearray()
        while len(body) <= limit:
            # read1 performs one buffered/socket read, allowing a deadline check
            # between reads rather than resetting a full timeout per packet.
            if connection.sock is not None:
                connection.sock.settimeout(remaining())
            else:
                remaining()
            chunk = response.read1(min(65536, limit + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
        return FetchResponse(
            response.status,
            {k.lower(): v for k, v in response.getheaders()},
            bytes(body),
        )
    except (OSError, http.client.HTTPException) as error:
        error.download_phase = phase
        raise
    finally:
        connection.close()
        if raw is not None:
            raw.close()


class SafeFetcher:
    def __init__(
        self,
        *,
        resolver: Callable = _resolve,
        transport: Callable = _transport,
        max_bytes: int = 5 * 1024 * 1024,
        interval: float = 1.0,
        timeout: float = 15.0,
    ):
        if max_bytes <= 0 or interval < 0 or timeout <= 0:
            raise ValueError("Invalid fetch limits")
        self.resolver, self.transport = resolver, transport
        self.max_bytes, self.interval, self.timeout = max_bytes, interval, timeout
        self._preferred_ips = {}

    def _request(self, url: str, last: list[float]) -> FetchResponse:
        host = urlsplit(url).hostname
        try:
            addresses = self.resolver(host)
        except OSError as error:
            raise FetchError(
                "DNS resolution failed", reason_code="dns_resolution_failed"
            ) from error
        if not addresses or any(
            not ipaddress.ip_address(a).is_global for a in addresses
        ):
            raise FetchError(
                "DNS returned a non-public address", reason_code="dns_non_public"
            )
        addresses = list(dict.fromkeys(addresses))
        preferred = self._preferred_ips.get(host)
        addresses.sort(key=lambda address: address != preferred)
        deadline = time.monotonic() + self.timeout
        attempts = []
        result = None
        for attempt in range(3):
            delay = self.interval - (time.monotonic() - last[0])
            if delay > 0:
                time.sleep(min(delay, max(0, deadline - time.monotonic())))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            address = addresses[attempt % len(addresses)]
            started = time.monotonic()
            try:
                result = self.transport(
                    url, address, self.max_bytes, min(5.0, remaining)
                )
                self._preferred_ips[host] = address
                break
            except (OSError, http.client.HTTPException) as error:
                attempts.append(
                    {
                        "url": url,
                        "address": address,
                        "error_type": type(error).__name__,
                        "phase": getattr(error, "download_phase", "connection"),
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                    }
                )
                if isinstance(error, ssl.SSLCertVerificationError):
                    raise FetchError(
                        "HTTPS certificate verification failed",
                        reason_code="https_download_failed",
                        attempts=attempts,
                    ) from error
            finally:
                last[0] = time.monotonic()
        if result is None:
            raise FetchError(
                "HTTPS download failed",
                reason_code="https_download_failed",
                attempts=attempts,
            )
        if len(result.body) > self.max_bytes:
            raise FetchError("Response byte budget exceeded")
        if result.headers.get("content-encoding", "identity").lower() not in (
            "",
            "identity",
        ):
            raise FetchError("Unsupported response encoding")
        return result

    def fetch(self, url: str) -> Snapshot:
        current = requested = validate_url(url)
        host = urlsplit(current).netloc
        last = [0.0]
        robots_url = f"https://{host}/robots.txt"
        robots = self._request(robots_url, last)
        if robots.status == 200:
            policy = RobotFileParser(robots_url)
            policy.parse(robots.body.decode("utf-8", errors="replace").splitlines())
        elif robots.status == 404:
            policy = None
        else:
            raise FetchError(
                "robots policy unavailable; import stopped",
                reason_code="robots_unavailable",
            )
        for _ in range(4):
            if policy is not None and not policy.can_fetch("DocGuard", current):
                raise FetchError(
                    "robots policy disallows this page", reason_code="robots_disallowed"
                )
            response = self._request(current, last)
            if response.status in (301, 302, 303, 307, 308):
                target = validate_url(
                    urljoin(current, response.headers.get("location", ""))
                )
                if urlsplit(target).netloc != host:
                    raise FetchError("Redirect leaves approved host")
                current = target
                continue
            if response.status != 200:
                raise FetchError(f"Page returned HTTP {response.status}")
            if response.headers.get("content-type", "").split(";")[
                0
            ].strip().lower() not in ("text/html", "application/xhtml+xml"):
                raise FetchError("Response is not HTML", reason_code="not_html")
            return Snapshot(
                requested,
                current,
                datetime.now(UTC).isoformat(),
                hashlib.sha256(response.body).hexdigest(),
                response.body,
                robots_url,
                robots.status,
                hashlib.sha256(robots.body).hexdigest(),
            )
        raise FetchError("Redirect limit exceeded")
