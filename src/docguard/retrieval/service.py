"""Version-aware dense retrieval service and S3 smoke-query runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ragbits.core.vector_stores.base import VectorStoreOptions, VectorStoreResult

from docguard.fs_safety import (
    ensure_directory,
    validate_directory,
    validate_regular_file,
)
from docguard.retrieval.config import load_dense_config
from docguard.retrieval.e5_embedder import E5Embedder
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.qdrant_factory import create_qdrant_client
from docguard.retrieval.qdrant_store import DocGuardQdrantStore

_VERSIONS = frozenset({"17.11", "18.6"})
_SMOKE_FIELDS = frozenset(
    {"schema_version", "query_id", "query", "version", "top_k"}
)


class DenseRetrievalError(RuntimeError):
    """Raised when stored retrieval data violates the frozen S3 contract."""


class RetrievalStore(Protocol):
    async def retrieve(
        self, text: str, options: VectorStoreOptions | None = None
    ) -> list[VectorStoreResult]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DenseSearchHit:
    rank: int
    chunk_id: str
    score: float
    postgresql_version: str
    topic: str
    section_path: tuple[str, ...]
    canonical_url: str
    anchor: str
    text: str


class DenseRetriever:
    """Lazy retrieval facade that keeps the Dense ablation switch side-effect free."""

    def __init__(self, store_factory: Callable[[], RetrievalStore]) -> None:
        self._store_factory = store_factory
        self._store: RetrievalStore | None = None

    def _get_store(self) -> RetrievalStore:
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    async def search(
        self,
        query: str,
        version: str | None,
        top_k: int,
        dense_enabled: bool = True,
    ) -> list[DenseSearchHit]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if version is not None and version not in _VERSIONS:
            raise ValueError("version must be null, 17.11, or 18.6")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if type(dense_enabled) is not bool:
            raise ValueError("dense_enabled must be a boolean")
        if not dense_enabled:
            return []

        where = None if version is None else {"postgresql_version": version}
        results = await self._get_store().retrieve(
            query.strip(), VectorStoreOptions(k=top_k, where=where)
        )
        mapped: list[tuple[float, str, dict[str, Any], str]] = []
        for result in results:
            metadata = result.entry.metadata
            score = float(result.score)
            if not math.isfinite(score):
                raise DenseRetrievalError("retrieval score must be finite")
            result_version = metadata.get("postgresql_version")
            if version is not None and result_version != version:
                raise DenseRetrievalError("retrieval result contains version leakage")
            chunk_id = metadata.get("chunk_id")
            if not isinstance(chunk_id, str):
                raise DenseRetrievalError("retrieval result is missing chunk_id")
            if result.entry.text is None:
                raise DenseRetrievalError("retrieval result is missing text")
            mapped.append((score, chunk_id, metadata, result.entry.text))

        mapped.sort(key=lambda item: (-item[0], item[1]))
        hits = []
        for rank, (score, chunk_id, metadata, text) in enumerate(mapped, start=1):
            section_path = metadata.get("section_path")
            if not isinstance(section_path, list) or not all(
                isinstance(item, str) for item in section_path
            ):
                raise DenseRetrievalError("retrieval result has invalid section_path")
            try:
                hit = DenseSearchHit(
                    rank=rank,
                    chunk_id=chunk_id,
                    score=score,
                    postgresql_version=str(metadata["postgresql_version"]),
                    topic=str(metadata["topic"]),
                    section_path=tuple(section_path),
                    canonical_url=str(metadata["canonical_url"]),
                    anchor=str(metadata["anchor"]),
                    text=text,
                )
            except KeyError as error:
                raise DenseRetrievalError(
                    f"retrieval result is missing metadata: {error.args[0]}"
                ) from error
            hits.append(hit)
        return hits

    async def close(self) -> None:
        if self._store is not None:
            await self._store.close()
            self._store = None


def create_default_retriever(
    *,
    index_root: Path = Path(".qdrant/s3-dense-v1"),
    config_path: Path = Path("configs/retrieval/s3-dense-v1.json"),
    model_root: Path = Path("data/models/s3-e5-small"),
    model_manifest_path: Path = Path("data/manifests/s3-embedding-model.json"),
    tokenizer_path: Path = Path(
        "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json"
    ),
    qdrant_url: str | None = None,
) -> DenseRetriever:
    config = load_dense_config(config_path)

    def factory() -> RetrievalStore:
        qdrant_path = index_root / "qdrant"
        if qdrant_url is None:
            validate_directory(qdrant_path)
        assets = verify_model_assets(model_root, model_manifest_path)
        embedder = E5Embedder(assets, tokenizer_path, config)
        return DocGuardQdrantStore(
            create_qdrant_client(path=qdrant_path)
            if qdrant_url is None
            else create_qdrant_client(url=qdrant_url),
            config.collection_name,
            embedder,
            vector_size=config.model.embedding_dimension,
        )

    return DenseRetriever(factory)


def _load_smoke_queries(path: Path) -> list[dict[str, Any]]:
    validate_regular_file(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != _SMOKE_FIELDS:
            raise ValueError(f"invalid smoke-query fields at line {line_number}")
        if row["schema_version"] != 1:
            raise ValueError(f"invalid schema_version at line {line_number}")
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not query_id or query_id in seen:
            raise ValueError(f"invalid or duplicate query_id at line {line_number}")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError(f"invalid query at line {line_number}")
        if row["version"] is not None and row["version"] not in _VERSIONS:
            raise ValueError(f"invalid version at line {line_number}")
        if row["top_k"] != 5:
            raise ValueError(f"top_k must be 5 at line {line_number}")
        seen.add(query_id)
        rows.append(row)
    if not rows:
        raise ValueError("smoke-query dataset must not be empty")
    return rows


def _jsonl_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


async def run_smoke(
    query_path: Path,
    output_path: Path,
    retriever: DenseRetriever | Any,
) -> int:
    rows = _load_smoke_queries(query_path)
    output_rows = []
    for row in rows:
        hits = await retriever.search(
            row["query"], row["version"], row["top_k"], dense_enabled=True
        )
        output_rows.append({**row, "hits": [asdict(hit) for hit in hits]})

    ensure_directory(output_path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.writelines(_jsonl_line(row) for row in output_rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(output_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--version", choices=sorted(_VERSIONS))
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--dense-enabled", action=argparse.BooleanOptionalAction, default=True)
    search.add_argument(
        "--index-root", type=Path, default=Path(".qdrant/s3-dense-v1")
    )
    smoke = subparsers.add_parser("run-smoke")
    smoke.add_argument("--queries", type=Path, default=Path("data/eval/s3-smoke-v1.jsonl"))
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument(
        "--index-root", type=Path, default=Path(".qdrant/s3-dense-v1")
    )
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    retriever = create_default_retriever(index_root=args.index_root)
    try:
        if args.command == "search":
            hits = await retriever.search(
                args.query, args.version, args.top_k, args.dense_enabled
            )
            print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, sort_keys=True))
        else:
            count = await run_smoke(args.queries, args.output, retriever)
            print(json.dumps({"query_count": count, "output": str(args.output)}, sort_keys=True))
    finally:
        await retriever.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_main_async(args))
    except (DenseRetrievalError, OSError, ValueError, json.JSONDecodeError) as error:
        parser = _parser()
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
