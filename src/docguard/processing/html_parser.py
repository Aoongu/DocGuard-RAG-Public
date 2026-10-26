"""Deterministic structural parsing for frozen PostgreSQL HTML pages."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator

from lxml import etree, html

from docguard.ingestion.models import SnapshotRecord
from docguard.processing.models import DocumentRecord, SectionBlock, make_document_id

_HEADING_TAGS = frozenset({f"h{level}" for level in range(1, 7)})
_SEMANTIC_CALLOUTS = frozenset({"note", "tip", "warning", "caution", "important"})
_DOCBOOK_SECTION_CLASSES = frozenset(
    {
        "appendix",
        "chapter",
        "part",
        "preface",
        "refentry",
        "refsynopsisdiv",
    }
)
_NOISE_XPATH = """
.//*[self::script or self::style or self::noscript
or contains(concat(' ', normalize-space(@class), ' '), ' navheader ')
or contains(concat(' ', normalize-space(@class), ' '), ' navfooter ')
or contains(concat(' ', normalize-space(@class), ' '), ' toc ')
or contains(concat(' ', normalize-space(@class), ' '), ' id_link ')]
"""


class HtmlParseError(ValueError):
    """Raised when a frozen page cannot be converted without ambiguity."""


def _error(snapshot: SnapshotRecord, message: str) -> HtmlParseError:
    return HtmlParseError(f"{snapshot.raw_path.name}: {message}")


def _tag(element: etree._Element) -> str:
    if not isinstance(element.tag, str):
        return ""
    return element.tag.rsplit("}", 1)[-1].lower()


def _classes(element: etree._Element) -> frozenset[str]:
    return frozenset(element.get("class", "").split())


def _normalize_inline(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\xa0", " "))
    return re.sub(r"\s+", " ", value).strip()


def _normalize_code(value: str) -> str:
    value = unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = [line.rstrip() for line in value.replace("\xa0", " ").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    if slug:
        return slug
    return "section-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _inside(element: etree._Element, tags: frozenset[str]) -> bool:
    return any(_tag(ancestor) in tags for ancestor in element.iterancestors())


def _callout_kind(element: etree._Element) -> str | None:
    for ancestor in element.iterancestors():
        matches = _classes(ancestor) & _SEMANTIC_CALLOUTS
        if matches:
            return min(matches)
    return None


def _heading_anchor(element: etree._Element, title: str) -> str:
    explicit = element.get("id")
    if explicit:
        return explicit
    for ancestor in element.iterancestors():
        if _tag(ancestor) == "body":
            break
        ancestor_id = ancestor.get("id")
        if ancestor_id:
            return ancestor_id
        classes = _classes(ancestor)
        is_numbered_section = any(
            name.startswith(("sect", "refsect")) for name in classes
        )
        if "refnamediv" not in classes and (
            classes & _DOCBOOK_SECTION_CLASSES or is_numbered_section
        ):
            break
    return _slug(title)


def _list_text(element: etree._Element) -> str:
    ordered = _tag(element) == "ol"
    lines: list[str] = []
    for index, item in enumerate(element.xpath("./li"), start=1):
        text = _normalize_inline(item.text_content())
        if text:
            marker = f"{index}." if ordered else "-"
            lines.append(f"{marker} {text}")
    return "\n".join(lines)


def _table_text(element: etree._Element) -> str:
    rows: list[str] = []
    for row in element.xpath(".//tr"):
        cells = [
            _normalize_inline(cell.text_content()) for cell in row.xpath("./th|./td")
        ]
        if cells and any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _semantic_elements(body: etree._Element) -> Iterator[etree._Element]:
    for element in body.iterdescendants():
        tag = _tag(element)
        is_paragraph = tag == "p" and not _inside(
            element, frozenset({"li", "td", "th", "pre"})
        )
        is_list = tag in {"ul", "ol"} and not _inside(
            element, frozenset({"ul", "ol"})
        )
        is_table = tag == "table" and not _inside(element, frozenset({"table"}))
        if is_paragraph or is_list or tag == "pre" or is_table:
            yield element


def parse_postgresql_html(
    html_bytes: bytes,
    snapshot: SnapshotRecord,
    *,
    max_input_bytes: int = 8 * 1024 * 1024,
    max_dom_nodes: int = 200_000,
) -> tuple[DocumentRecord, tuple[SectionBlock, ...]]:
    """Parse one hash-pinned PostgreSQL page into ordered semantic blocks."""
    if not html_bytes:
        raise _error(snapshot, "input is empty")
    if len(html_bytes) > max_input_bytes or len(html_bytes) != snapshot.size_bytes:
        raise _error(snapshot, "input size does not match the frozen snapshot")
    actual_sha256 = hashlib.sha256(html_bytes).hexdigest()
    if actual_sha256 != snapshot.content_sha256:
        raise _error(snapshot, "input SHA-256 does not match the frozen snapshot")
    try:
        parser = html.HTMLParser(
            encoding="utf-8", no_network=True, recover=True, remove_comments=True
        )
        root = html.fromstring(html_bytes, parser=parser)
    except (etree.ParserError, UnicodeError, ValueError) as error:
        raise _error(snapshot, "HTML parsing failed") from error
    nodes = list(root.iter())
    if len(nodes) > max_dom_nodes:
        raise _error(snapshot, f"DOM node limit exceeded: {len(nodes)}")
    bodies = root.xpath("//body[@id='docContent']")
    if not bodies:
        raise _error(snapshot, "required body#docContent container is missing")
    body = bodies[0]
    title_nodes = root.xpath("//head/title")
    title = _normalize_inline(title_nodes[0].text_content()) if title_nodes else ""
    if not title:
        first_heading = next(
            (node for node in body.iterdescendants() if _tag(node) in _HEADING_TAGS),
            None,
        )
        title = (
            _normalize_inline(first_heading.text_content())
            if first_heading is not None
            else ""
        )
    if not title:
        raise _error(snapshot, "document title is missing")
    toc_items: list[str] = []
    seen_toc_items: set[str] = set()
    for toc in body.xpath(
        ".//*[contains(concat(' ', normalize-space(@class), ' '), ' toc ')]"
    ):
        for link in toc.xpath(".//a"):
            item = _normalize_inline(link.text_content())
            if item and item not in seen_toc_items:
                seen_toc_items.add(item)
                toc_items.append(item)
    for noise in list(body.xpath(_NOISE_XPATH)):
        parent = noise.getparent()
        if parent is not None:
            parent.remove(noise)

    current_anchor = _slug(title)
    section_by_level: dict[int, str] = {}
    used_heading_anchors: set[str] = set()
    blocks: list[SectionBlock] = []
    semantic_elements = set(_semantic_elements(body))
    for element in body.iterdescendants():
        tag = _tag(element)
        if tag in _HEADING_TAGS and _callout_kind(element) is None:
            heading = _normalize_inline(element.text_content())
            if not heading:
                continue
            level = int(tag[1])
            for stale_level in tuple(section_by_level):
                if stale_level >= level:
                    del section_by_level[stale_level]
            section_by_level[level] = heading
            anchor = _heading_anchor(element, heading)
            if anchor in used_heading_anchors:
                raise _error(snapshot, f"anchor collision: {anchor.lower()}")
            used_heading_anchors.add(anchor)
            current_anchor = anchor
            continue
        if element not in semantic_elements:
            continue
        section_path = tuple(section_by_level[level] for level in sorted(section_by_level))
        if not section_path:
            section_path = (title,)
        if tag == "p":
            text = _normalize_inline(element.text_content())
            callout = _callout_kind(element)
            if callout and text:
                text = f"{callout.title()}: {text}"
            kind = "paragraph"
        elif tag in {"ul", "ol"}:
            text = _list_text(element)
            kind = "list"
        elif tag == "pre":
            text = _normalize_code("".join(element.itertext()))
            kind = "code"
        else:
            text = _table_text(element)
            kind = "table"
        if text:
            blocks.append(
                SectionBlock(
                    kind=kind,  # type: ignore[arg-type]
                    anchor=current_anchor,
                    section_path=section_path,
                    text=text,
                )
            )
    if not blocks and toc_items:
        section_path = tuple(section_by_level[level] for level in sorted(section_by_level))
        blocks.append(
            SectionBlock(
                kind="list",
                anchor=current_anchor,
                section_path=section_path or (title,),
                text="\n".join(f"- {item}" for item in toc_items),
            )
        )
    if not blocks:
        raise _error(snapshot, "no semantic block was extracted")
    document = DocumentRecord(
        document_id=make_document_id(
            snapshot.postgresql_version, snapshot.canonical_url
        ),
        postgresql_version=snapshot.postgresql_version,
        topic=snapshot.topic,
        canonical_url=snapshot.canonical_url,
        title=title,
        content_sha256=snapshot.content_sha256,
    )
    return document, tuple(blocks)
