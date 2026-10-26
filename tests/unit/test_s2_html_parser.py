import hashlib
from pathlib import Path, PurePosixPath

import pytest

from docguard.ingestion.models import SnapshotRecord
from docguard.processing.html_parser import HtmlParseError, parse_postgresql_html
from docguard.processing.models import make_document_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "s2" / "postgresql_sample.html"
URL = "https://www.postgresql.org/docs/17/sql-select.html"


def _snapshot(payload: bytes, **changes: object) -> SnapshotRecord:
    values: dict[str, object] = {
        "schema_version": 1,
        "postgresql_version": "17.11",
        "topic": "sql_commands",
        "archive_path": PurePosixPath(
            "data/archives/postgresql-17.11-docs.tar.gz"
        ),
        "archive_sha256": "a" * 64,
        "archive_member": PurePosixPath(
            "postgresql-17.11/doc/src/sgml/html/sql-select.html"
        ),
        "raw_path": PurePosixPath("data/raw/postgresql-17.11/sql-select.html"),
        "canonical_url": URL,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    values.update(changes)
    return SnapshotRecord(**values)  # type: ignore[arg-type]


def test_parser_extracts_ordered_normalized_semantic_blocks() -> None:
    payload = FIXTURE.read_bytes().replace(b"\n", b"\r\n")

    document, blocks = parse_postgresql_html(payload, _snapshot(payload))

    assert document.document_id == make_document_id("17.11", URL)
    assert document.title == "Café & SELECT"
    assert document.content_sha256 == hashlib.sha256(payload).hexdigest()
    assert [
        (block.kind, block.anchor, block.section_path, block.text)
        for block in blocks
    ] == [
        ("paragraph", "overview", ("Overview",), "Intro text & Café."),
        ("paragraph", "overview", ("Overview",), "Note: Keep this."),
        (
            "list",
            "overview",
            ("Overview",),
            "- first item\n- second item & more",
        ),
        (
            "paragraph",
            "details",
            ("Overview", "Details & Setup"),
            "Configure shared_buffers.",
        ),
        (
            "paragraph",
            "examples",
            ("Overview", "Details & Setup", "Examples"),
            "Malformed markup",
        ),
        (
            "code",
            "examples",
            ("Overview", "Details & Setup", "Examples"),
            "SELECT  1;\n  SELECT 2;",
        ),
        (
            "table",
            "examples",
            ("Overview", "Details & Setup", "Examples"),
            "Name | Value\nalpha | 1\nbeta | 2",
        ),
    ]
    combined = "\n".join(block.text for block in blocks)
    assert "Previous Home Next" not in combined
    assert "window.noise" not in combined


def test_parser_uses_immediate_heading_parent_id() -> None:
    payload = b"""<html><head><title>T</title></head><body id='docContent'>
    <section id='parent-anchor'><h2>Parent anchored</h2></section>
    <p>Body.</p></body></html>"""

    _, blocks = parse_postgresql_html(payload, _snapshot(payload))

    assert blocks[0].anchor == "parent-anchor"
    assert blocks[0].section_path == ("Parent anchored",)


def test_parser_uses_nearest_nested_section_ancestor_id() -> None:
    payload = b"""<html><head><title>T</title></head><body id='docContent'>
    <section id='REAL-ANCHOR'><div class='titlepage'><div><h2>Nested title</h2></div></div>
    <p>Body.</p></section></body></html>"""

    _, blocks = parse_postgresql_html(payload, _snapshot(payload))

    assert blocks[0].anchor == "REAL-ANCHOR"
    assert blocks[0].section_path == ("Nested title",)


def test_parser_does_not_reuse_refentry_id_for_unanchored_subsections() -> None:
    payload = b"""<html><head><title>ALTER GROUP</title></head><body id='docContent'>
    <div class='refentry' id='SQL-ALTERGROUP'>
      <div class='refnamediv'><h2>ALTER GROUP</h2><p>change a role name</p></div>
      <div class='refsynopsisdiv'><h2>Synopsis</h2><pre>ALTER GROUP name</pre></div>
    </div></body></html>"""

    _, blocks = parse_postgresql_html(payload, _snapshot(payload))

    assert [(block.anchor, block.section_path) for block in blocks] == [
        ("SQL-ALTERGROUP", ("ALTER GROUP",)),
        ("synopsis", ("Synopsis",)),
    ]


def test_parser_preserves_toc_only_navigation_page_as_one_list_block() -> None:
    payload = b"""<html><head><title>Chapter 12</title></head>
    <body id='docContent'><div id='TEXTSEARCH'><h2>Chapter 12</h2>
    <div class='toc'><a href='one.html'>1. Intro</a>
    <a href='two.html'>2. Setup</a></div></div></body></html>"""

    _, blocks = parse_postgresql_html(payload, _snapshot(payload))

    assert [(block.kind, block.anchor, block.section_path, block.text) for block in blocks] == [
        ("list", "TEXTSEARCH", ("Chapter 12",), "- 1. Intro\n- 2. Setup")
    ]


@pytest.mark.parametrize(
    ("payload", "kwargs", "message"),
    [
        (b"", {}, "sql-select.html.*empty"),
        (b"12345", {"max_input_bytes": 4}, "sql-select.html.*size"),
        (b"<html><body><p>x</p></body></html>", {}, "sql-select.html.*body"),
        (
            b"<html><body id='docContent'><h1>Only title</h1></body></html>",
            {},
            "sql-select.html.*block",
        ),
        (
            b"<html><body id='docContent'><h1>T</h1><p>x</p><p>y</p></body></html>",
            {"max_dom_nodes": 4},
            "sql-select.html.*node",
        ),
    ],
)
def test_parser_fails_closed_for_invalid_input(
    payload: bytes, kwargs: dict[str, int], message: str
) -> None:
    snapshot = _snapshot(payload if payload else b"non-empty snapshot")
    with pytest.raises(HtmlParseError, match=message):
        parse_postgresql_html(payload, snapshot, **kwargs)


def test_parser_rejects_snapshot_content_mismatch() -> None:
    payload = b"<html><body id='docContent'><h1>T</h1><p>x</p></body></html>"
    snapshot = _snapshot(payload, content_sha256="0" * 64)

    with pytest.raises(HtmlParseError, match="sql-select.html.*SHA-256"):
        parse_postgresql_html(payload, snapshot)


def test_parser_rejects_fallback_anchor_collision() -> None:
    payload = b"""<html><head><title>T</title></head><body id='docContent'>
    <h2>Repeated</h2><p>one</p><h2>Repeated</h2><p>two</p>
    </body></html>"""

    with pytest.raises(HtmlParseError, match="sql-select.html.*anchor.*repeated"):
        parse_postgresql_html(payload, _snapshot(payload))

