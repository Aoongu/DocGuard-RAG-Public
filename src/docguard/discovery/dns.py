"""Opt-in public DoH resolution, independent of local Fake-IP DNS.

Bootstrap IPs and TLS hostnames are fixed. No environment proxies, redirects,
custom endpoints or fallback to system DNS. Both address families are checked.
"""

import http.client
import ipaddress
import json
import os
import socket
import ssl
import threading
import time
from urllib.parse import urlencode

from docguard.discovery.fetch import FetchError, _resolve


def _query(host, kind):
    return _query_at(host, kind, "cloudflare-dns.com", "1.1.1.1", "/dns-query")


def _query_alidns(host, kind):
    return _query_at(host, kind, "dns.alidns.com", "223.5.5.5", "/resolve")


def _query_at(host, kind, tls_host, address, path):
    connection = http.client.HTTPSConnection(tls_host, timeout=6)
    raw = None
    try:
        raw = socket.create_connection((address, 443), timeout=6)
        connection.sock = ssl.create_default_context().wrap_socket(
            raw, server_hostname=tls_host
        )
        connection.request(
            "GET",
            path + "?" + urlencode({"name": host, "type": kind}),
            headers={"Accept": "application/dns-json"},
        )
        response = connection.getresponse()
        body = response.read(65537)
        if response.status == 429 or response.status >= 500:
            raise OSError("DoH provider unavailable")
        if (
            response.status != 200
            or len(body) > 65536
            or response.getheader("Content-Type", "").split(";")[0].strip().lower()
            not in ("application/dns-json", "application/json")
        ):
            raise ValueError("Invalid DoH response")
        return json.loads(body)
    finally:
        connection.close()
        if raw is not None:
            raw.close()


class DohResolver:
    def __init__(self, *, query=None, provider=None):
        self.provider = (
            (provider or os.environ.get("DOCGUARD_DOH_PROVIDER", "auto"))
            .strip()
            .lower()
        )
        if self.provider not in ("auto", "cloudflare", "alidns"):
            raise ValueError("DOCGUARD_DOH_PROVIDER must be auto, cloudflare or alidns")
        self.query = query or _query
        available = {"cloudflare": _query, "alidns": _query_alidns}
        self._providers = (
            [("injected", query)]
            if query is not None
            else list(available.items())
            if self.provider == "auto"
            else [(self.provider, available[self.provider])]
        )
        self.last_provider = None
        self._cache = {}
        self._lock = threading.Lock()

    def __call__(self, host):
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            if not literal.is_global:
                raise FetchError(
                    "DNS returned a non-public address", reason_code="dns_non_public"
                )
            return [str(literal)]
        with self._lock:
            cached = self._cache.get(host)
            if cached and cached[0] > time.monotonic():
                return list(cached[1])
        error = None
        providers = sorted(
            self._providers, key=lambda pair: pair[0] != self.last_provider
        )
        for provider, query in providers:
            try:
                addresses, ttls = self._lookup(host, query)
            except FetchError:
                raise
            except (OSError, http.client.HTTPException) as failure:
                error = failure
                continue
            except (ValueError, TypeError, KeyError, AttributeError) as failure:
                raise FetchError(
                    "Invalid DoH response", reason_code="dns_resolution_failed"
                ) from failure
            self.last_provider = provider
            break
        else:
            raise FetchError(
                "DoH providers unreachable", reason_code="dns_resolution_failed"
            ) from error
        result = sorted(addresses)
        with self._lock:
            if len(self._cache) >= 128:
                self._cache.pop(next(iter(self._cache)))
            self._cache[host] = (time.monotonic() + min([60, *ttls]), result)
        return list(result)

    @staticmethod
    def _lookup(host, query):
        addresses, ttls = set(), []
        for kind in (1, 28):
            payload = query(host, kind)
            if type(payload.get("Status")) is not int or payload["Status"] != 0:
                raise ValueError("DNS lookup failed")
            questions = payload.get("Question", [])
            if isinstance(questions, dict):
                questions = [questions]
            if (
                len(questions) != 1
                or questions[0].get("name", "").lower().rstrip(".")
                != host.lower().rstrip(".")
                or questions[0].get("type") != kind
            ):
                raise ValueError("Mismatched DNS question")
            for answer in payload.get("Answer", []):
                if answer.get("type") not in (1, 5, 28):
                    continue
                ttl = answer.get("TTL", 0)
                ttls.append(ttl if type(ttl) is int and ttl >= 0 else 0)
                if answer["type"] == 5:
                    continue
                address = ipaddress.ip_address(answer["data"])
                if not address.is_global:
                    raise FetchError(
                        "DNS returned a non-public address",
                        reason_code="dns_non_public",
                    )
                addresses.add(str(address))
        if not addresses:
            raise ValueError("No public addresses")
        return addresses, ttls


def configured_resolver():
    mode = os.environ.get("DOCGUARD_DOCUMENT_DNS", "system").strip().lower()
    if mode == "system":
        return _resolve
    if mode == "doh":
        return DohResolver()
    raise ValueError("DOCGUARD_DOCUMENT_DNS must be system or doh")
