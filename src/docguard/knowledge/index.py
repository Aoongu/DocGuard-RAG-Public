"""Generic SQLite BM25 and optional Qdrant indexes for one immutable snapshot."""

import json
import math
import re
import sqlite3
import unicodedata
import uuid
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient, models

from docguard.fs_safety import validate_regular_file
from docguard.knowledge.models import Chunk
from docguard.retrieval.bm25_index import bm25_score
from docguard.retrieval.technical_tokenizer import tokenize_technical


def terms(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tokenize_technical(normalized) + tuple(
        re.findall(r"[\u3400-\u9fff]", normalized)
    )


def build_bm25(path: Path, chunks: list[Chunk]) -> None:
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript(
            "CREATE TABLE chunks(id TEXT PRIMARY KEY, payload TEXT, length INTEGER); CREATE TABLE postings(term TEXT, id TEXT, tf INTEGER, PRIMARY KEY(term,id));"
        )
        for chunk in chunks:
            tokens = terms(chunk.text)
            db.execute(
                "INSERT INTO chunks VALUES (?,?,?)",
                (
                    chunk.chunk_id,
                    json.dumps(asdict(chunk), ensure_ascii=False),
                    len(tokens),
                ),
            )
            db.executemany(
                "INSERT INTO postings VALUES (?,?,?)",
                [
                    (term, chunk.chunk_id, tf)
                    for term, tf in sorted(Counter(tokens).items())
                ],
            )


def read_chunks(path: Path) -> dict[str, Chunk]:
    validate_regular_file(path)
    with closing(
        sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    ) as db:
        output = {}
        for chunk_id, raw in db.execute("SELECT id,payload FROM chunks ORDER BY id"):
            data = json.loads(raw)
            data["section_path"] = tuple(data["section_path"])
            output[chunk_id] = Chunk(**data)
        return output


def search_bm25(path: Path, query: str, limit: int) -> list[tuple[str, float]]:
    validate_regular_file(path)
    with closing(
        sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    ) as db:
        count, average = db.execute(
            "SELECT COUNT(*),AVG(length) FROM chunks"
        ).fetchone()
        if not count or not average:
            return []
        scores: dict[str, float] = {}
        for term in sorted(set(terms(query))):
            rows = db.execute(
                "SELECT p.id,p.tf,c.length FROM postings p JOIN chunks c ON p.id=c.id WHERE p.term=?",
                (term,),
            ).fetchall()
            for chunk_id, tf, length in rows:
                scores[chunk_id] = scores.get(chunk_id, 0) + bm25_score(
                    tf, len(rows), length, count, average, k1=1.2, b=0.75
                )
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def validate_vectors(vectors, count: int, dimension: int | None = None) -> int:
    if len(vectors) != count or not vectors:
        raise ValueError("Dense vector count mismatch")
    size = dimension or len(vectors[0])
    if size <= 0 or any(
        len(v) != size
        or any(not math.isfinite(float(x)) for x in v)
        or not any(float(x) != 0 for x in v)
        for v in vectors
    ):
        raise ValueError(
            "Dense vectors must be finite, nonzero and dimensionally consistent"
        )
    return size


def build_dense(path: Path, chunks: list[Chunk], vectors) -> int:
    dimension = validate_vectors(vectors, len(chunks))
    client = QdrantClient(path=str(path))
    try:
        client.create_collection(
            "documents",
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )
        client.upsert(
            "documents",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                    vector=v,
                    payload={
                        "chunk_id": c.chunk_id,
                        "knowledge_base_id": c.knowledge_base_id,
                    },
                )
                for c, v in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )
    finally:
        client.close()
    return dimension


def search_dense(path: Path, vector, kb_id: str, limit: int) -> list[tuple[str, float]]:
    client = QdrantClient(path=str(path))
    try:
        hits = client.query_points(
            "documents",
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="knowledge_base_id", match=models.MatchValue(value=kb_id)
                    )
                ]
            ),
            limit=limit,
        ).points
        return [(h.payload["chunk_id"], h.score) for h in hits]
    finally:
        client.close()
