from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient
from ragbits.core.vector_stores.base import VectorStoreEntry, VectorStoreOptions

from docguard.retrieval.qdrant_store import (
    DOCGUARD_NAMESPACE,
    DocGuardQdrantStore,
    point_id,
)


class FakeE5Embedder:
    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        mapping = {
            "alpha primary": [1.0, 0.0, 0.0],
            "alpha secondary": [0.9, 0.1, 0.0],
            "beta": [0.0, 1.0, 0.0],
            "gamma": [0.0, 0.0, 1.0],
        }
        return [mapping[text] for text in texts]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 1
        return [[1.0, 0.0, 0.0]]


def _entry(index: int, text: str, version: str) -> VectorStoreEntry:
    chunk_id = f"chk_{index:064x}"
    return VectorStoreEntry(
        id=point_id(chunk_id),
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "postgresql_version": version,
            "topic": "sql_commands",
            "section_path": ["SELECT", "Examples"],
            "canonical_url": (
                f"https://www.postgresql.org/docs/{version.split('.')[0]}/sql-select.html"
            ),
            "anchor": "SQL-SELECT-EXAMPLES",
            "ordinal": index,
            "content_sha256": f"{index + 10:064x}",
        },
    )


@pytest.mark.asyncio
async def test_qdrant_store_persists_payload_and_filters_versions(tmp_path: Path) -> None:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = DocGuardQdrantStore(
        client,
        "test_collection",
        FakeE5Embedder(),
        vector_size=3,
    )
    entries = [
        _entry(1, "alpha primary", "17.11"),
        _entry(2, "alpha secondary", "17.11"),
        _entry(3, "beta", "18.6"),
        _entry(4, "gamma", "18.6"),
    ]
    try:
        await store.store(entries)

        assert await store.count() == 4
        assert await store.version_counts() == {"17.11": 2, "18.6": 2}
        all_results = await store.retrieve("alpha", VectorStoreOptions(k=4))
        version_17 = await store.retrieve(
            "alpha", VectorStoreOptions(k=5, where={"postgresql_version": "17.11"})
        )
        version_18 = await store.retrieve(
            "alpha", VectorStoreOptions(k=5, where={"postgresql_version": "18.6"})
        )

        assert [result.entry.metadata["chunk_id"] for result in all_results[:2]] == [
            entries[0].metadata["chunk_id"],
            entries[1].metadata["chunk_id"],
        ]
        assert [result.score for result in all_results] == sorted(
            (result.score for result in all_results), reverse=True
        )
        assert {result.entry.metadata["postgresql_version"] for result in version_17} == {
            "17.11"
        }
        assert {result.entry.metadata["postgresql_version"] for result in version_18} == {
            "18.6"
        }
        assert version_17[0].entry.metadata["section_path"] == ["SELECT", "Examples"]
    finally:
        await store.close()


def test_point_id_is_stable_uuid5() -> None:
    from uuid import uuid5

    chunk_id = "chk_" + "a" * 64

    assert point_id(chunk_id) == uuid5(DOCGUARD_NAMESPACE, chunk_id)


@pytest.mark.asyncio
async def test_qdrant_store_rejects_incomplete_payload(tmp_path: Path) -> None:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = DocGuardQdrantStore(
        client,
        "test_collection",
        FakeE5Embedder(),
        vector_size=3,
    )
    incomplete = VectorStoreEntry(
        id=point_id("chk_" + "a" * 64),
        text="alpha primary",
        metadata={"chunk_id": "chk_" + "a" * 64},
    )
    try:
        with pytest.raises(ValueError, match="metadata fields"):
            await store.store([incomplete])
        assert await store.count() == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_qdrant_store_rejects_unsupported_filter(tmp_path: Path) -> None:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = DocGuardQdrantStore(
        client,
        "test_collection",
        FakeE5Embedder(),
        vector_size=3,
    )
    try:
        with pytest.raises(ValueError, match="postgresql_version"):
            await store.retrieve("alpha", VectorStoreOptions(where={"topic": "sql_commands"}))
    finally:
        await store.close()
