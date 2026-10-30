"""Ground provisional model source assessments in already fetched page links."""

from urllib.parse import urldefrag, urljoin

from lxml import html

from docguard.discovery.fetch import validate_url


def discovery_links(snapshot):
    """Expose real anchor labels and prefer broad documentation entry points."""
    root = html.fromstring(
        snapshot.body,
        parser=html.HTMLParser(encoding="utf-8", no_network=True, remove_comments=True),
    )
    entries = {}
    for anchor in root.xpath("//a[@href]")[:1000]:
        try:
            url = validate_url(
                urldefrag(urljoin(snapshot.final_url, anchor.get("href")))[0]
            )
        except ValueError:
            continue
        title = " ".join(anchor.text_content().split())[:160]
        if url not in entries or not entries[url]["title"]:
            entries[url] = {"url": url, "title": title}

    def rank(link):
        label = link["title"].casefold()
        priority = (
            0
            if label
            in (
                "user guide",
                "user's guide",
                "documentation",
                "docs",
                "getting started",
                "用户指南",
                "使用指南",
                "入门",
                "文档",
                "home",
            )
            else 1
        )
        return priority, link["url"]

    return sorted(entries.values(), key=rank)[:200]


def source_links(snapshot):
    root = html.fromstring(
        snapshot.body,
        parser=html.HTMLParser(encoding="utf-8", no_network=True, remove_comments=True),
    )
    links = set()
    for href in root.xpath("//a/@href")[:1000]:
        try:
            links.add(validate_url(urldefrag(urljoin(snapshot.final_url, href))[0]))
        except ValueError:
            continue
    return sorted(links)[:200]


def source_assessment(store, task_id, candidate, authorities, reason):
    """A real link supports a proposal, never proves independent ownership."""
    if not reason or not authorities or len(authorities) > 5:
        raise ValueError("Source assessment requires a rationale and supporting pages")
    document = store.snapshot(task_id, candidate["candidate_id"])
    targets = {document.requested_url, document.final_url}
    proof = []
    for authority in authorities:
        if authority["candidate_id"] == candidate["candidate_id"]:
            raise ValueError("A page cannot corroborate itself")
        snapshot = store.snapshot(task_id, authority["candidate_id"])
        if snapshot.final_url == document.final_url:
            raise ValueError("Supporting page resolves to the same document")
        linked = targets.intersection(source_links(snapshot))
        if not linked:
            raise ValueError("Supporting page must actually link to the document")
        proof.append(
            {
                "url": snapshot.final_url,
                "content_sha256": snapshot.content_sha256,
                "linked_document_urls": sorted(linked),
            }
        )
    return {
        "method": "linked_pages_model_assessment",
        "reason": reason,
        "document_url": document.final_url,
        "document_content_sha256": document.content_sha256,
        "authority_pages": proof,
        "requires_user_confirmation": True,
        "independent_ownership_verified": False,
    }
