"""Ragbits VectorStore implementation for the DocGuard dense index."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import AsyncQdrantClient, models
from ragbits.core.vector_stores.base import (
    VectorStore,
    VectorStoreEntry,
    VectorStoreOptions,
    VectorStoreResult,
    WhereQuery,
)

DOCGUARD_NAMESPACE = uuid5(NAMESPACE_URL, "https://github.com/Aoongu/DocGuard-RAG")
_CHUNK_ID_RE = re.compile(r"chk_[0-9a-f]{64}")
_VERSIONS = frozenset({"17.11", "18.6"})
_METADATA_FIELDS = frozenset(
    {
        "chunk_id",
        "postgresql_version",
        "topic",
        "section_path",
        "canonical_url",
        "anchor",
        "ordinal",
        "content_sha256",
    }
)


class PassageQueryEmbedder(Protocol):
    async def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


def point_id(chunk_id: str) -> UUID:
    """Map a stable Chunk ID to Qdrant's UUID identifier space."""
    if _CHUNK_ID_RE.fullmatch(chunk_id) is None:
        raise ValueError("chunk_id must be a full prefixed SHA-256")
    return uuid5(DOCGUARD_NAMESPACE, chunk_id)


class DocGuardQdrantStore(VectorStore[VectorStoreOptions]):
    """Persistent dense vector store with a fixed payload/filter contract."""

    options_cls = VectorStoreOptions
    _VECTOR_NAME = "dense"

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        embedder: PassageQueryEmbedder,
        *,
        vector_size: int,
        default_options: VectorStoreOptions | None = None,
    ) -> None:
        super().__init__(default_options=default_options)
        if not collection_name:
            raise ValueError("collection_name must be non-empty")
        if type(vector_size) is not int or vector_size <= 0:
            raise ValueError("vector_size must be positive")
        self._client = client
        self._collection_name = collection_name
        self._embedder = embedder
        self._vector_size = vector_size

    @staticmethod
    def _validate_entry(entry: VectorStoreEntry) -> None:
        if entry.text is None or not entry.text.strip():
            raise ValueError("vector store entry text must be non-empty")
        if set(entry.metadata) != _METADATA_FIELDS:
            raise ValueError("vector store metadata fields do not match the S3 schema")
        chunk_id = entry.metadata["chunk_id"]
        if entry.id != point_id(chunk_id):
            raise ValueError("vector store entry UUID does not match chunk_id")
        if entry.metadata["postgresql_version"] not in _VERSIONS:
            raise ValueError("unsupported PostgreSQL version")

    async def _ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection_name):
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    self._VECTOR_NAME: models.VectorParams(
                        size=self._vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

    async def store(self, entries: list[VectorStoreEntry]) -> None:
        if not entries:
            return
        for entry in entries:
            self._validate_entry(entry)
        vectors = await self._embedder.embed_passages(
            [entry.text for entry in entries if entry.text is not None]
        )
        if len(vectors) != len(entries):
            raise ValueError("embedder returned an unexpected vector count")
        if any(len(vector) != self._vector_size for vector in vectors):
            raise ValueError("embedder returned an unexpected vector dimension")
        await self._ensure_collection()
        points = [
            models.PointStruct(
                id=str(entry.id),
                vector={self._VECTOR_NAME: vector},
                payload=entry.model_dump(exclude_none=True, mode="json"),
            )
            for entry, vector in zip(entries, vectors, strict=True)
        ]
        await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    @staticmethod
    def _filter(where: WhereQuery | None) -> models.Filter | None:
        if where is None:
            return None
        if set(where) != {"postgresql_version"} or where["postgresql_version"] not in _VERSIONS:
            raise ValueError("only postgresql_version 17.11/18.6 filtering is supported")
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.postgresql_version",
                    match=models.MatchValue(value=where["postgresql_version"]),
                )
            ]
        )

    async def retrieve(
        self, text: str, options: VectorStoreOptions | None = None
    ) -> list[VectorStoreResult]:
        merged = self.default_options | options if options else self.default_options
        if not text.strip():
            raise ValueError("query text must be non-empty")
        query_filter = self._filter(merged.where)
        if not await self._client.collection_exists(self._collection_name):
            return []
        vector = (await self._embedder.embed_queries([text]))[0]
        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            using=self._VECTOR_NAME,
            query_filter=query_filter,
            limit=merged.k,
            score_threshold=merged.score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for point in response.points:
            if point.payload is None:
                raise ValueError("Qdrant result is missing payload")
            results.append(
                VectorStoreResult(
                    entry=VectorStoreEntry.model_validate(point.payload),
                    score=float(point.score),
                    vector=[],
                )
            )
        return results

    async def count(self, version: str | None = None) -> int:
        if not await self._client.collection_exists(self._collection_name):
            return 0
        where = None if version is None else {"postgresql_version": version}
        result = await self._client.count(
            collection_name=self._collection_name,
            count_filter=self._filter(where),
            exact=True,
        )
        return result.count

    async def version_counts(self) -> dict[str, int]:
        return {version: await self.count(version) for version in sorted(_VERSIONS)}

    async def list(
        self, where: WhereQuery | None = None, limit: int | None = None, offset: int = 0
    ) -> list[VectorStoreEntry]:
        if not await self._client.collection_exists(self._collection_name):
            return []
        count = await self.count() if limit is None else limit
        if count <= 0:
            return []
        records, _ = await self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=self._filter(where),
            limit=count,
            offset=offset or None,
            with_payload=True,
            with_vectors=False,
        )
        return [
            VectorStoreEntry.model_validate(record.payload)
            for record in records
            if record.payload is not None
        ]

    async def remove(self, ids: list[UUID]) -> None:
        if not ids or not await self._client.collection_exists(self._collection_name):
            return
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[str(item) for item in ids]),
            wait=True,
        )

    async def collection_schema(self) -> dict[str, object] | None:
        if not await self._client.collection_exists(self._collection_name):
            return None
        info = await self._client.get_collection(self._collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or self._VECTOR_NAME not in vectors:
            raise ValueError("Qdrant collection is missing the dense vector")
        dense = vectors[self._VECTOR_NAME]
        return {
            "vector_name": self._VECTOR_NAME,
            "size": dense.size,
            "distance": dense.distance.value,
        }

    async def close(self) -> None:
        await self._client.close()


def entries_in_batches(
    entries: Sequence[VectorStoreEntry], batch_size: int
) -> list[list[VectorStoreEntry]]:
    """Split entries deterministically for bounded embedding/upsert batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(entries[index : index + batch_size]) for index in range(0, len(entries), batch_size)]
