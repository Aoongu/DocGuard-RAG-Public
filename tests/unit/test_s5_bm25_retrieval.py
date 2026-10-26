from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docguard.processing.models import ChunkRecord, make_chunk_id, make_document_id
from docguard.processing.storage import chunk_dict
from docguard.retrieval import bm25_index as bm25_module
from docguard.retrieval.bm25_index import BM25Retriever, build_bm25_index


def _chunk(version: str, page: str, text: str) -> ChunkRecord:
    major = version.split(".", 1)[0]
    url = f"https://www.postgresql.org/docs/{major}/{page}.html"
    return ChunkRecord(
        schema_version=1,
        chunk_id=make_chunk_id("s2-chunk-v1", version, url, "main", 0),
        document_id=make_document_id(version, url),
        postgresql_version=version,
        topic="installation_configuration",
        canonical_url=url,
        anchor="main",
        section_path=(page,),
        ordinal=0,
        text=text,
        token_count=max(1, len(text.split())),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        chunking_version="s2-chunk-v1",
    )


def _index(tmp_path: Path) -> tuple[Path, Path, list[ChunkRecord]]:
    chunks = [
        _chunk("17.11", "wal", "wal_level replica SQLSTATE 23505 duplicate key"),
        _chunk("17.11", "hba", "pg_hba.conf client authentication error"),
        _chunk("18.6", "wal", "wal_level logical logical logical"),
        _chunk("18.6", "connections", "--max-connections client login"),
    ]
    payload = "".join(
        json.dumps(chunk_dict(item), ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in chunks
    ).encode()
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_bytes(payload)
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/retrieval/s5-hybrid-v1.json").read_text(encoding="utf-8")
    )
    config["chunks_sha256"] = hashlib.sha256(payload).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    index_root = tmp_path / "index"
    build_bm25_index(chunks_path, index_root, config_path=config_path)
    return index_root, config_path, chunks


@pytest.mark.parametrize(
    ("query", "page"),
    [
        ("wal_level", "/wal.html"),
        ("pg_hba.conf", "/hba.html"),
        ("--max-connections", "/connections.html"),
        ("SQLSTATE 23505", "/wal.html"),
        ("duplicate key error", "/wal.html"),
    ],
)
async def test_exact_technical_queries_retrieve_containing_chunk(
    tmp_path: Path, query: str, page: str
) -> None:
    index_root, config_path, _ = _index(tmp_path)
    retriever = BM25Retriever(index_root, config_path=config_path)
    try:
        hits = await retriever.search(query, None, 4)
    finally:
        await retriever.close()

    assert hits
    assert hits[0].canonical_url.endswith(page)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


async def test_version_filter_changes_scoped_statistics_without_leakage(
    tmp_path: Path,
) -> None:
    index_root, config_path, _ = _index(tmp_path)
    retriever = BM25Retriever(index_root, config_path=config_path)
    try:
        v17 = await retriever.search("wal_level", "17.11", 5)
        v18 = await retriever.search("wal_level", "18.6", 5)
    finally:
        await retriever.close()

    assert [hit.postgresql_version for hit in v17] == ["17.11"]
    assert [hit.postgresql_version for hit in v18] == ["18.6"]
    assert v17[0].score != v18[0].score


async def test_score_order_and_chunk_id_tie_break_are_deterministic(
    tmp_path: Path,
) -> None:
    index_root, config_path, _ = _index(tmp_path)
    retriever = BM25Retriever(index_root, config_path=config_path)
    try:
        hits = await retriever.search("client", None, 5)
    finally:
        await retriever.close()

    assert hits == sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id))
    assert len({hit.chunk_id for hit in hits}) == len(hits)


async def test_non_finite_aggregate_score_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_root, config_path, _ = _index(tmp_path)
    retriever = BM25Retriever(index_root, config_path=config_path)
    monkeypatch.setattr(bm25_module, "bm25_score", lambda *args, **kwargs: float("nan"))
    try:
        with pytest.raises(bm25_module.BM25IndexError, match="finite"):
            await retriever.search("wal_level", None, 5)
    finally:
        await retriever.close()


async def test_disabled_route_does_not_open_missing_sqlite(tmp_path: Path) -> None:
    retriever = BM25Retriever(tmp_path / "missing")

    assert await retriever.search("wal_level", "17.11", 5, bm25_enabled=False) == []


@pytest.mark.parametrize(
    ("query", "version", "top_k", "enabled"),
    [
        ("", None, 5, True),
        ("wal", "19.0", 5, True),
        ("wal", None, 0, True),
        ("wal", None, 5, 1),
    ],
)
async def test_invalid_search_arguments_fail(
    tmp_path: Path, query: str, version: str | None, top_k: int, enabled: bool
) -> None:
    retriever = BM25Retriever(tmp_path / "missing")

    with pytest.raises((TypeError, ValueError)):
        await retriever.search(query, version, top_k, bm25_enabled=enabled)
