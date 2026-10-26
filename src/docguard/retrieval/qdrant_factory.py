"""Create an explicit local or service-mode Qdrant client."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from qdrant_client import AsyncQdrantClient


def create_qdrant_client(
    *,
    path: Path | None = None,
    url: str | None = None,
    check_compatibility: bool = True,
) -> AsyncQdrantClient:
    if type(check_compatibility) is not bool:
        raise TypeError("check_compatibility must be a boolean")
    if (path is None) == (url is None):
        raise ValueError("exactly one Qdrant mode must be configured")
    if path is not None:
        return AsyncQdrantClient(path=str(path))
    assert url is not None
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Qdrant service URL must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Qdrant credentials must not be embedded in the URL")
    return AsyncQdrantClient(url=url, check_compatibility=check_compatibility)
