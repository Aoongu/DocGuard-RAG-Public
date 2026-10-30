"""Extract structured blocks from static HTML without inventing URL anchors."""

import hashlib
import re
import unicodedata
from urllib.parse import urljoin, urlsplit, urlunsplit

from lxml import etree, html

from docguard.knowledge.models import Block, Page


class ExtractionError(ValueError):
    """The page does not contain usable static documentation."""


def _inline(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def extract_html(payload: bytes, url: str) -> Page:
    if not payload or len(payload) > 5 * 1024 * 1024:
        raise ExtractionError("HTML is empty or oversized")
    try:
        root = html.fromstring(
            payload,
            parser=html.HTMLParser(
                encoding="utf-8", no_network=True, remove_comments=True
            ),
        )
    except (etree.ParserError, ValueError) as error:
        raise ExtractionError("HTML parsing failed") from error
    if sum(1 for _ in root.iter()) > 200_000:
        raise ExtractionError("HTML node budget exceeded")
    if root.xpath('//input[translate(@type,"PASSWORD","password")="password"]'):
        raise ExtractionError("Login page is not documentation")
    titles = root.xpath("//title")
    title = _inline(titles[0].text_content()) if titles else ""
    containers = root.xpath(
        '//main | //article | //body[@id="docContent"] | //*[@role="main"]'
    )
    if not containers:
        raise ExtractionError("No supported documentation container")
    body = containers[0]
    for node in list(
        body.xpath(".//script|.//style|.//noscript|.//nav|.//footer|.//aside|.//form")
    ):
        if node.getparent() is not None:
            node.drop_tree()
    headings: list[tuple[int, str]] = []
    anchor = None
    blocks: list[Block] = []
    for node in body.iterdescendants():
        if not isinstance(node.tag, str):
            continue
        tag = node.tag.lower()
        ancestor_tags = {
            str(p.tag).lower() for p in node.iterancestors() if p is not body
        }
        if ancestor_tags & {"pre", "table", "ul", "ol"}:
            continue
        if re.fullmatch(r"h[1-6]", tag):
            heading = _inline(node.text_content())
            if not heading:
                continue
            level = int(tag[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, heading))
            anchor = node.get("id")
            if anchor is None:
                anchor = next(
                    (
                        p.get("id")
                        for p in node.iterancestors()
                        if p is not body and p.get("id")
                    ),
                    None,
                )
            if not title:
                title = heading
            continue
        path = tuple(text for _, text in headings) or (title or "Document",)
        if tag == "p":
            text, kind = _inline(node.text_content()), "paragraph"
        elif tag == "pre":
            text = unicodedata.normalize(
                "NFC", node.text_content().replace("\r\n", "\n").replace("\r", "\n")
            ).strip("\n")
            kind = "code"
        elif tag in {"ul", "ol"}:
            text = "\n".join(
                f"{i}. {_inline(n.text_content())}"
                if tag == "ol"
                else f"- {_inline(n.text_content())}"
                for i, n in enumerate(node.xpath("./li"), 1)
            )
            kind = "list"
        elif tag == "table":
            text = "\n".join(
                " | ".join(_inline(c.text_content()) for c in row.xpath("./th|./td"))
                for row in node.xpath(".//tr")
            )
            kind = "table"
        else:
            continue
        if text.strip():
            blocks.append(Block(kind, anchor, path, text.strip()))
    if not blocks:
        raise ExtractionError("No document content; dynamic or empty page")
    links = set()
    for href in body.xpath(".//a/@href"):
        parsed = urlsplit(urljoin(url, href))
        if (
            parsed.scheme == "https"
            and parsed.netloc == urlsplit(url).netloc
            and not parsed.username
        ):
            links.add(urlunsplit(parsed._replace(fragment="")))
    return Page(
        url,
        title or blocks[0].section_path[0],
        hashlib.sha256(payload).hexdigest(),
        tuple(blocks),
        tuple(sorted(links)),
    )
