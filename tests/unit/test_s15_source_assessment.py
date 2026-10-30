import hashlib
from types import SimpleNamespace

import pytest

from docguard.discovery.fetch import Snapshot
from docguard.discovery.sources import source_assessment, source_links


def snapshot(url, body, final_url=None):
    body = body.encode()
    return Snapshot(
        url,
        final_url or url,
        "2026-10-28T00:00:00Z",
        hashlib.sha256(body).hexdigest(),
        body,
        url + "robots.txt",
        404,
        hashlib.sha256(b"").hexdigest(),
    )


def test_discovery_navigation_preserves_titles_and_prioritizes_user_guide():
    from docguard.discovery.sources import discovery_links

    page = snapshot(
        "https://docs.example/development/extending.html",
        "<nav>"
        + "".join(f'<a href="/api/a{i}">Thing {i}</a>' for i in range(20))
        + '<a href="/user_guide/index.html">User Guide</a>'
        + '<a href="https://127.0.0.1/">Documentation</a></nav>',
    )
    links = discovery_links(page)
    assert links[0] == {
        "url": "https://docs.example/user_guide/index.html",
        "title": "User Guide",
    }
    assert all("127.0.0.1" not in link["url"] for link in links)


def test_source_links_include_real_cross_domain_navigation_but_reject_unsafe_urls():
    page = snapshot(
        "https://project.example/",
        '<nav><a href="https://docs.example/guide#intro">Docs</a><a href="http://127.0.0.1/">Local</a><a href="https://127.0.0.1/">Local</a><a href="javascript:alert(1)">Script</a></nav>',
    )
    assert source_links(page) == ["https://docs.example/guide"]


@pytest.mark.parametrize("mode", ["self", "missing", "unlinked", "same_redirect"])
def test_source_assessment_cannot_use_invented_unread_or_self_corroboration(mode):
    url = "https://docs.example/guide"
    pages = {
        "doc": snapshot(url, "<p>Docs</p>"),
        "home": snapshot("https://project.example/", "<p>No link</p>"),
    }
    authority = {"candidate_id": "home"}
    if mode == "self":
        authority = {"candidate_id": "doc"}
    elif mode == "missing":
        pages.pop("home")
    elif mode == "same_redirect":
        pages["home"] = snapshot(
            "https://project.example/", f'<a href="{url}">Docs</a>', final_url=url
        )

    def load(task_id, candidate_id):
        if candidate_id not in pages:
            raise LookupError("Page not read")
        return pages[candidate_id]

    with pytest.raises((ValueError, LookupError)):
        source_assessment(
            SimpleNamespace(snapshot=load),
            "task",
            {"candidate_id": "doc"},
            [authority],
            "Model thinks this is official",
        )


def test_real_link_yields_only_a_provisional_assessment():
    url = "https://docs.example/guide"
    pages = {
        "doc": snapshot(url, "<p>Docs</p>"),
        "home": snapshot(
            "https://project.example/", f'<nav><a href="{url}">Docs</a></nav>'
        ),
    }
    store = SimpleNamespace(snapshot=lambda task_id, candidate_id: pages[candidate_id])
    proof = source_assessment(
        store,
        "task",
        {"candidate_id": "doc"},
        [{"candidate_id": "home"}],
        "Project homepage links to the same product docs",
    )
    assert proof["requires_user_confirmation"] is True
    assert proof["independent_ownership_verified"] is False
    assert proof["document_content_sha256"] == pages["doc"].content_sha256
