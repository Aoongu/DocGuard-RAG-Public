"""Build and verify the persistent S3 dense index."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from ragbits.core.vector_stores.base import VectorStoreEntry

from docguard.fs_safety import (
    FileSafetyError,
    atomic_rename_directory_noreplace,
    create_stage,
    ensure_directory,
    exclusive_lock,
    validate_directory,
    validate_regular_file,
    write_stage_file,
)
from docguard.processing.models import ChunkRecord
from docguard.processing.storage import stable_json_bytes
from docguard.retrieval.config import DenseConfig, load_dense_config
from docguard.retrieval.e5_embedder import E5Embedder
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.qdrant_store import (
    DocGuardQdrantStore,
    PassageQueryEmbedder,
    entries_in_batches,
    point_id,
)


class DenseIndexError(RuntimeError):
    """Raised when an S3 dense index cannot be built or verified."""


class DenseIndexConflictError(DenseIndexError):
    """Raised when an existing index differs from the frozen inputs."""


@dataclass(frozen=True, slots=True)
class DenseIndexResult:
    status: str
    point_count: int
    version_counts: dict[str, int]


def _sha256(path: Path) -> str:
    try:
        validate_regular_file(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, FileSafetyError) as error:
        raise DenseIndexError(f"unable to hash input file: {path}") from error


def _load_chunks(path: Path, config: DenseConfig) -> list[ChunkRecord]:
    if _sha256(path) != config.chunks_sha256:
        raise DenseIndexError("chunks SHA-256 does not match the S3 config")
    records = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError("chunk record must be an object")
            if isinstance(data.get("section_path"), list):
                data["section_path"] = tuple(data["section_path"])
            record = ChunkRecord(**data)
            if record.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id at line {line_number}")
            if record.chunking_version != config.chunking_version:
                raise ValueError(f"unexpected chunking_version at line {line_number}")
            seen.add(record.chunk_id)
            records.append(record)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise DenseIndexError(f"invalid chunks JSONL: {path}: {error}") from error
    if not records:
        raise DenseIndexError("chunks JSONL must not be empty")
    return records


def _entry(chunk: ChunkRecord) -> VectorStoreEntry:
    return VectorStoreEntry(
        id=point_id(chunk.chunk_id),
        text=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "postgresql_version": chunk.postgresql_version,
            "topic": chunk.topic,
            "section_path": list(chunk.section_path),
            "canonical_url": chunk.canonical_url,
            "anchor": chunk.anchor,
            "ordinal": chunk.ordinal,
            "content_sha256": chunk.content_sha256,
        },
    )


def _build_info(
    config: DenseConfig,
    *,
    config_path: Path,
    model_manifest_path: Path,
    point_count: int,
    version_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "index_version": config.index_version,
        "collection_name": config.collection_name,
        "vector_name": "dense",
        "vector_size": config.model.embedding_dimension,
        "distance": config.distance,
        "point_count": point_count,
        "version_counts": version_counts,
        "config_sha256": _sha256(config_path),
        "s2_manifest_sha256": config.s2_manifest_sha256,
        "chunks_sha256": config.chunks_sha256,
        "model_manifest_sha256": _sha256(model_manifest_path),
        "model_name": config.model.name,
        "model_revision": config.model.revision,
    }


def _default_embedder(
    config: DenseConfig,
    model_root: Path,
    model_manifest_path: Path,
    tokenizer_path: Path,
) -> E5Embedder:
    assets = verify_model_assets(model_root, model_manifest_path)
    return E5Embedder(assets, tokenizer_path, config)


async def _verify_open_index(
    chunks: list[ChunkRecord],
    output_root: Path,
    config: DenseConfig,
    expected_info: dict[str, Any],
    embedder: PassageQueryEmbedder,
) -> DenseIndexResult:
    try:
        validate_directory(output_root)
        children = {path.name: path for path in output_root.iterdir()}
        if set(children) != {"build-info.json", "qdrant"}:
            raise DenseIndexConflictError("dense index output file set mismatch")
        validate_regular_file(children["build-info.json"])
        validate_directory(children["qdrant"])
        actual_info = json.loads(children["build-info.json"].read_text(encoding="utf-8"))
    except DenseIndexConflictError:
        raise
    except (OSError, FileSafetyError, json.JSONDecodeError) as error:
        raise DenseIndexConflictError("unsafe or invalid dense index build-info") from error
    if actual_info != expected_info:
        raise DenseIndexConflictError("dense index build-info differs from frozen inputs")

    client = AsyncQdrantClient(path=str(children["qdrant"]))
    store = DocGuardQdrantStore(
        client,
        config.collection_name,
        embedder,
        vector_size=config.model.embedding_dimension,
    )
    try:
        schema = await store.collection_schema()
        expected_schema = {
            "vector_name": "dense",
            "size": config.model.embedding_dimension,
            "distance": config.distance,
        }
        if schema != expected_schema:
            raise DenseIndexConflictError("Qdrant collection schema mismatch")
        point_count = await store.count()
        version_counts = await store.version_counts()
        if point_count != len(chunks) or sum(version_counts.values()) != point_count:
            raise DenseIndexConflictError("Qdrant point counts do not match chunks")
        entries = await store.list(limit=point_count)
        expected_metadata = {chunk.chunk_id: _entry(chunk).metadata for chunk in chunks}
        actual_metadata = {
            str(entry.metadata.get("chunk_id")): entry.metadata for entry in entries
        }
        if actual_metadata != expected_metadata:
            raise DenseIndexConflictError("Qdrant payload differs from chunks")
    finally:
        await store.close()
    return DenseIndexResult("verified", point_count, version_counts)


async def build_index(
    chunks_path: Path,
    output_root: Path,
    *,
    config_path: Path = Path("configs/retrieval/s3-dense-v1.json"),
    model_root: Path = Path("data/models/s3-e5-small"),
    model_manifest_path: Path = Path("data/manifests/s3-embedding-model.json"),
    tokenizer_path: Path = Path(
        "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json"
    ),
    embedder: PassageQueryEmbedder | None = None,
) -> DenseIndexResult:
    config = load_dense_config(config_path)
    chunks = _load_chunks(chunks_path, config)
    actual_embedder = embedder or _default_embedder(
        config, model_root, model_manifest_path, tokenizer_path
    )
    version_counts = {
        version: sum(chunk.postgresql_version == version for chunk in chunks)
        for version in ("17.11", "18.6")
    }
    expected_info = _build_info(
        config,
        config_path=config_path,
        model_manifest_path=model_manifest_path,
        point_count=len(chunks),
        version_counts=version_counts,
    )
    parent = output_root.parent
    ensure_directory(parent)
    with exclusive_lock(parent, output_root.name + ".build"):
        if os.path.lexists(output_root):
            verified = await _verify_open_index(
                chunks, output_root, config, expected_info, actual_embedder
            )
            return DenseIndexResult("skipped", verified.point_count, verified.version_counts)
        stage, stage_identity = create_stage(parent, f".{output_root.name}.stage-")
        store: DocGuardQdrantStore | None = None
        try:
            qdrant_path = stage / "qdrant"
            client = AsyncQdrantClient(path=str(qdrant_path))
            store = DocGuardQdrantStore(
                client,
                config.collection_name,
                actual_embedder,
                vector_size=config.model.embedding_dimension,
            )
            entries = [_entry(chunk) for chunk in chunks]
            for batch in entries_in_batches(entries, config.batch_size):
                await store.store(batch)
            point_count = await store.count()
            actual_counts = await store.version_counts()
            if point_count != len(chunks) or actual_counts != version_counts:
                raise DenseIndexError("built Qdrant counts do not match chunks")
            await store.close()
            store = None
            write_stage_file(stage, "build-info.json", stable_json_bytes(expected_info))
            validate_directory(stage, stage_identity)
            if {path.name for path in stage.iterdir()} != {"build-info.json", "qdrant"}:
                raise DenseIndexError("dense index staging file set mismatch")
            atomic_rename_directory_noreplace(stage, output_root)
        except Exception as error:
            if store is not None:
                await store.close()
            raise DenseIndexError(
                f"staging retained for manual inspection: {stage}: {error}"
            ) from error
    return DenseIndexResult("built", len(chunks), version_counts)


async def verify_index(
    chunks_path: Path,
    output_root: Path,
    *,
    config_path: Path = Path("configs/retrieval/s3-dense-v1.json"),
    model_root: Path = Path("data/models/s3-e5-small"),
    model_manifest_path: Path = Path("data/manifests/s3-embedding-model.json"),
    tokenizer_path: Path = Path(
        "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json"
    ),
    embedder: PassageQueryEmbedder | None = None,
) -> DenseIndexResult:
    config = load_dense_config(config_path)
    chunks = _load_chunks(chunks_path, config)
    actual_embedder = embedder or _default_embedder(
        config, model_root, model_manifest_path, tokenizer_path
    )
    version_counts = {
        version: sum(chunk.postgresql_version == version for chunk in chunks)
        for version in ("17.11", "18.6")
    }
    expected_info = _build_info(
        config,
        config_path=config_path,
        model_manifest_path=model_manifest_path,
        point_count=len(chunks),
        version_counts=version_counts,
    )
    try:
        return await _verify_open_index(
            chunks, output_root, config, expected_info, actual_embedder
        )
    except DenseIndexConflictError:
        raise
    except Exception as error:
        raise DenseIndexConflictError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--chunks", type=Path, default=Path("data/processed/s2-chunk-v1/chunks.jsonl")
        )
        child.add_argument(
            "--output-root", type=Path, default=Path(".qdrant/s3-dense-v1")
        )
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    function = build_index if args.command == "build" else verify_index
    result = await function(args.chunks, args.output_root)
    print(
        json.dumps(
            {
                "status": result.status,
                "point_count": result.point_count,
                "version_counts": result.version_counts,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_main_async(args))
    except (DenseIndexError, FileSafetyError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
