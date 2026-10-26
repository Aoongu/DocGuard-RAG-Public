from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import pytest
from ragbits.core.vector_stores.base import VectorStoreEntry, VectorStoreOptions

from docguard.engineering.regression import RankedHit, compare_rankings
from docguard.retrieval.qdrant_factory import create_qdrant_client
from docguard.retrieval.qdrant_store import DocGuardQdrantStore, point_id


class FixtureEmbedder:
    _vectors: ClassVar[dict[str, list[float]]] = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "gamma": [0.0, 0.0, 1.0],
        "question": [1.0, 0.5, 0.0],
    }

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]


def _entry(label: str, suffix: str) -> VectorStoreEntry:
    chunk_id = "chk_" + suffix * 64
    return VectorStoreEntry(
        id=point_id(chunk_id),
        text=label,
        metadata={
            "chunk_id": chunk_id,
            "postgresql_version": "18.6",
            "topic": "fixture",
            "section_path": ["Fixture"],
            "canonical_url": "https://www.postgresql.org/docs/18/fixture.html",
            "anchor": label,
            "ordinal": 0,
            "content_sha256": suffix * 64,
        },
    )


async def _ranking(store: DocGuardQdrantStore) -> list[RankedHit]:
    rows = await store.retrieve("question", VectorStoreOptions(k=3))
    return [
        RankedHit(str(row.entry.metadata["chunk_id"]), float(row.score)) for row in rows
    ]


@pytest.mark.asyncio
async def test_local_qdrant_persists_across_client_restart(tmp_path: Path) -> None:
    path = tmp_path / "qdrant"
    collection = "s11_local_persistence"
    first = DocGuardQdrantStore(
        create_qdrant_client(path=path), collection, FixtureEmbedder(), vector_size=3
    )
    await first.store([_entry("alpha", "a")])
    await first.close()
    second = DocGuardQdrantStore(
        create_qdrant_client(path=path), collection, FixtureEmbedder(), vector_size=3
    )
    assert await second.count() == 1
    await second.close()


@pytest.mark.asyncio
async def test_local_and_service_qdrant_have_equivalent_top_k(tmp_path: Path) -> None:
    url = os.environ.get("S11_QDRANT_URL")
    if not url:
        pytest.skip("set S11_QDRANT_URL to run the Compose service parity check")
    collection = f"s11_parity_{uuid4().hex}"
    local = DocGuardQdrantStore(
        create_qdrant_client(path=tmp_path / "local"),
        collection,
        FixtureEmbedder(),
        vector_size=3,
    )
    remote_client = create_qdrant_client(url=url)
    remote = DocGuardQdrantStore(
        remote_client, collection, FixtureEmbedder(), vector_size=3
    )
    entries = [_entry("alpha", "a"), _entry("beta", "b"), _entry("gamma", "c")]
    try:
        await local.store(entries)
        await remote.store(entries)
        compare_rankings(await _ranking(local), await _ranking(remote), tolerance=1e-6)
    finally:
        await local.close()
        await remote_client.delete_collection(collection)
        await remote.close()
