import hashlib

import pytest

from docguard.discovery.extract import ExtractionError, extract_html

PAGE = b"""<html><head><title>Redis persistence</title></head><body>
<nav>NOISE</nav><main><h1>Persistence</h1><p>Redis supports RDB snapshots.</p>
<h2 id="aof">Append only file</h2><p>AOF records write operations.</p>
<pre>appendonly yes\nappendfsync everysec</pre>
<table><tr><th>Option</th><th>Value</th></tr><tr><td>appendonly</td><td>yes</td></tr></table>
<a href="../backup/">Backup</a><script>INJECTION</script></main><footer>NOISE</footer></body></html>"""


def test_extract_keeps_structure_and_only_real_anchors():
    page = extract_html(PAGE, "https://redis.io/docs/persistence/")
    assert page.title == "Redis persistence"
    assert page.content_sha256 == hashlib.sha256(PAGE).hexdigest()
    assert page.blocks[0].anchor is None
    assert page.blocks[1].anchor == "aof"
    assert page.blocks[1].section_path == ("Persistence", "Append only file")
    assert {block.kind for block in page.blocks} == {"paragraph", "code", "table"}
    assert "appendfsync everysec" in page.blocks[2].text
    assert page.links == ("https://redis.io/docs/backup/",)
    assert all("NOISE" not in b.text and "INJECTION" not in b.text for b in page.blocks)


def test_extract_supports_article_and_unicode():
    page = extract_html(
        "<title>Python</title><article><h1>配置</h1><p>读取配置文件。</p></article>".encode(),
        "https://docs.python.org/3/config.html",
    )
    assert page.blocks[0].text == "读取配置文件。"
    assert page.blocks[0].anchor is None


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"<html><nav>menu</nav></html>",
        b"<main><h1>Empty</h1></main>",
        b'<main><form><input type="password"></form><p>Sign in</p></main>',
    ],
)
def test_extract_rejects_empty_navigation_and_login_pages(body):
    with pytest.raises(ExtractionError):
        extract_html(body, "https://example.com/docs/")


def test_duplicate_nested_blocks_are_not_emitted_twice():
    page = extract_html(
        b"<article><h1>X</h1><ul><li><p>one</p></li><li>two</li></ul><pre><code>x = 1</code></pre></article>",
        "https://example.com/docs/",
    )
    assert [b.kind for b in page.blocks] == ["list", "code"]
    assert page.blocks[0].text.count("one") == 1


def test_extract_does_not_follow_external_canonical_url():
    page = extract_html(
        b'<head><title>X</title><link rel="canonical" href="https://evil.test/x"></head><main><p>Document text.</p></main>',
        "https://example.com/docs/",
    )
    assert page.canonical_url == "https://example.com/docs/"
