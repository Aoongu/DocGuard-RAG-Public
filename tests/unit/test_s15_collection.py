import hashlib

import pytest

from docguard.discovery.fetch import Snapshot


def snap(url, body, final_url=None):
    body = body.encode()
    return Snapshot(
        url,
        final_url or url,
        "2026-10-29T00:00:00Z",
        hashlib.sha256(body).hexdigest(),
        body,
        url + "robots.txt",
        404,
        hashlib.sha256(b"").hexdigest(),
    )


def test_collection_follows_all_chapters_and_keeps_scope_and_cache():
    from docguard.discovery.collection import collect, scope_for_entry

    root = "https://docs.example/3/guide/index.html"
    pages = {
        root: snap(
            root,
            "<main>"
            + "".join(f'<a href="chapter{i}.html">Chapter {i}</a>' for i in range(25))
            + '<a href="../api/index.html">API</a><a href="https://evil.example/a.html">external</a><a href="%2e%2e/api/x.html">escape</a><a href="file.pdf">pdf</a></main>',
        )
    }
    pages.update(
        {
            f"https://docs.example/3/guide/chapter{i}.html": snap(
                f"https://docs.example/3/guide/chapter{i}.html",
                '<main>Content<a href="index.html">Home</a></main>',
            )
            for i in range(25)
        }
    )
    calls, cache = [], {}

    def load(url):
        if url not in cache:
            calls.append(url)
            cache[url] = pages[url]
        return cache[url]

    result = collect(scope_for_entry(root), load)
    assert len(result) == 26
    assert len(calls) == 26
    assert len(collect(scope_for_entry(root), load)) == 26
    assert len(calls) == 26


def test_collection_rejects_truncation_and_redirect_outside_scope():
    from docguard.discovery.collection import CollectionError, collect, scope_for_entry

    root = "https://docs.example/3/guide/index.html"
    scope = scope_for_entry(root)
    scope["max_pages"] = 1
    with pytest.raises(CollectionError, match="page_limit"):
        collect(scope, lambda url: snap(url, '<a href="next.html">Next</a>'))
    with pytest.raises(CollectionError, match="redirect_outside_scope"):
        collect(
            scope_for_entry(root),
            lambda url: snap(url, "outside", "https://docs.example/2/guide/index.html"),
        )
    assert scope_for_entry("https://docs.example/3/guide/chapter.html") is None
