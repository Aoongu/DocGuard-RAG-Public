"""Bounded deterministic collection of a confirmed documentation directory."""

import hashlib
import time
from collections import deque
from urllib.parse import unquote, urldefrag, urljoin, urlsplit

from lxml import html

from docguard.discovery.fetch import validate_url

MAX_PAGES = 1000
MAX_BYTES = 512 * 1024 * 1024
MAX_SECONDS = 1800


class CollectionError(ValueError):
    """The complete approved collection could not be collected."""


def document_kind(url, title):
    text = (urlsplit(url).path + " " + title).casefold().replace("_", "-")
    if any(
        word in text
        for word in ("user-guide", "user guide", "manual", "用户指南", "使用指南")
    ):
        return "user_guide"
    if any(
        word in text
        for word in ("/reference/", "api reference", "api-reference", "api参考")
    ):
        return "api_reference"
    if any(
        word in text
        for word in (
            "getting-started",
            "getting started",
            "quickstart",
            "tutorial",
            "入门",
        )
    ):
        return "getting_started"
    return None


def scope_for_entry(url):
    url = validate_url(url)
    parts = urlsplit(url)
    path = parts.path
    if parts.query or any(
        segment in (".", "..") for segment in unquote(path).split("/")
    ):
        return None
    if path.endswith(("/index.html", "/index.htm")):
        prefix = path.rsplit("/", 1)[0] + "/"
    elif path.endswith("/"):
        prefix = path
    else:
        return None
    if prefix == "/":
        return None
    return {
        "root_url": url,
        "origin": f"https://{parts.netloc}",
        "path_prefix": prefix,
        "max_pages": MAX_PAGES,
        "max_bytes": MAX_BYTES,
        "max_seconds": MAX_SECONDS,
    }


def within_scope(url, scope):
    try:
        url = validate_url(url)
        parts = urlsplit(url)
        path = parts.path
        for _ in range(3):
            path = unquote(path)
        return (
            f"https://{parts.netloc}" == scope["origin"]
            and not parts.query
            and path.startswith(unquote(scope["path_prefix"]))
            and not any(segment in (".", "..") for segment in path.split("/"))
            and (
                path.endswith(("/", ".html", ".htm"))
                or "." not in path.rsplit("/", 1)[-1]
            )
        )
    except ValueError:
        return False


def collect(
    scope, load, *, progress=lambda value: None, check=lambda: None, elapsed=0.0
):
    """Load/cache every reachable HTML page; limits fail instead of truncating."""
    queue = deque([scope["root_url"]])
    scheduled = {scope["root_url"]}
    result, final_urls = {}, set()
    total, started = 0, time.monotonic()
    while queue:
        check()
        if elapsed + time.monotonic() - started > scope["max_seconds"]:
            raise CollectionError("collection_time_limit")
        url = queue.popleft()
        snapshot = load(url)
        if not within_scope(snapshot.final_url, scope):
            raise CollectionError("redirect_outside_scope")
        if hashlib.sha256(snapshot.body).hexdigest() != snapshot.content_sha256:
            raise CollectionError("collection_snapshot_mismatch")
        if snapshot.final_url in final_urls:
            continue
        final_urls.add(snapshot.final_url)
        total += len(snapshot.body)
        if total > scope["max_bytes"]:
            raise CollectionError("collection_byte_limit")
        if elapsed + time.monotonic() - started > scope["max_seconds"]:
            raise CollectionError("collection_time_limit")
        result[url] = snapshot
        root = html.fromstring(
            snapshot.body,
            parser=html.HTMLParser(
                encoding="utf-8", no_network=True, remove_comments=True
            ),
        )
        hrefs = root.xpath("//a/@href")
        if len(hrefs) > 10000:
            raise CollectionError("collection_link_limit")
        links = set()
        for href in hrefs:
            link = urldefrag(urljoin(snapshot.final_url, href))[0]
            if within_scope(link, scope):
                links.add(validate_url(link))
        for link in sorted(links):
            if link not in scheduled and link not in final_urls:
                if len(scheduled) >= scope["max_pages"]:
                    raise CollectionError("collection_page_limit")
                scheduled.add(link)
                queue.append(link)
        progress(
            {
                "phase": "collecting",
                "completed_pages": len(result),
                "discovered_pages": len(scheduled),
                "downloaded_bytes": total,
                "active_seconds": elapsed + time.monotonic() - started,
                "last_url": url,
            }
        )
    return result
