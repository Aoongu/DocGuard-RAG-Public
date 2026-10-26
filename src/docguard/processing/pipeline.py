"""Offline build, verification, and statistics CLI for the S2 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from docguard.fs_safety import (
    FileConflictError,
    FileSafetyError,
    atomic_rename_directory_noreplace,
    create_stage,
    ensure_directory,
    exclusive_lock,
    validate_directory,
    validate_regular_file,
    verify_stage,
    write_stage_file,
)
from docguard.ingestion.models import SnapshotRecord
from docguard.ingestion.snapshot import load_snapshot_manifest
from docguard.processing.chunker import chunk_document
from docguard.processing.config import load_chunking_config
from docguard.processing.html_parser import parse_postgresql_html
from docguard.processing.storage import (
    corpus_stats,
    jsonl_bytes,
    logical_sqlite_digest,
    logical_sqlite_digest_bytes,
    metadata_sqlite_bytes,
    stable_json_bytes,
)
from docguard.processing.token_budget import TokenBudget

_OUTPUT_FILES = frozenset(
    {"build-info.json", "chunks.jsonl", "metadata.sqlite3", "stats.json"}
)


class PipelineError(RuntimeError):
    """Raised when an S2 corpus cannot be built or verified."""


class PipelineConflictError(PipelineError):
    """Raised when existing processed output differs from the frozen inputs."""


@dataclass(frozen=True, slots=True)
class BuildResult:
    status: str
    document_count: int
    chunk_count: int


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise PipelineError(f"unable to read input file: {path}") from error


def _read_raw_files(raw_root: Path, records: Sequence[SnapshotRecord]) -> list[bytes]:
    expected_by_directory: dict[Path, set[str]] = {}
    payloads: list[bytes] = []
    for record in records:
        version_directory = raw_root / f"postgresql-{record.postgresql_version}"
        expected_by_directory.setdefault(version_directory, set()).add(record.raw_path.name)
    for directory, expected in expected_by_directory.items():
        try:
            validate_directory(directory)
            actual = {path.name for path in directory.iterdir()}
        except (OSError, FileSafetyError) as error:
            raise PipelineError(f"unsafe raw directory: {directory}") from error
        if actual != expected:
            raise PipelineError(f"raw file set mismatch: {directory}")
    for record in records:
        path = raw_root / f"postgresql-{record.postgresql_version}" / record.raw_path.name
        try:
            validate_regular_file(path)
            payload = path.read_bytes()
        except (OSError, FileSafetyError) as error:
            raise PipelineError(f"unsafe raw file: {path}") from error
        if len(payload) != record.size_bytes:
            raise PipelineError(f"raw size mismatch: {path}")
        if hashlib.sha256(payload).hexdigest() != record.content_sha256:
            raise PipelineError(f"raw SHA-256 mismatch: {path}")
        payloads.append(payload)
    return payloads


def _expected_artifacts(
    raw_root: Path,
    *,
    config_path: Path,
    tokenizer_manifest_path: Path,
    snapshot_manifest_path: Path,
) -> tuple[dict[str, bytes], int, int]:
    records = tuple(
        sorted(
            load_snapshot_manifest(snapshot_manifest_path),
            key=lambda item: (item.postgresql_version, item.canonical_url),
        )
    )
    payloads = _read_raw_files(raw_root, records)
    config = load_chunking_config(config_path)
    tokenizer_repository_root = tokenizer_manifest_path.resolve().parents[2]
    budget = TokenBudget.from_manifest(
        tokenizer_manifest_path, tokenizer_repository_root
    )
    documents = []
    chunks = []
    for record, payload in zip(records, payloads, strict=True):
        document, blocks = parse_postgresql_html(payload, record)
        documents.append(document)
        chunks.extend(chunk_document(document, blocks, config, budget))
    chunks_payload = jsonl_bytes(chunks)
    stats_payload = stable_json_bytes(corpus_stats(documents, chunks))
    base_info = {
        "schema_version": 1,
        "chunking_version": config.chunking_version,
        "snapshot_manifest_sha256": _sha256_path(snapshot_manifest_path),
        "config_sha256": _sha256_path(config_path),
        "tokenizer_manifest_sha256": _sha256_path(tokenizer_manifest_path),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks_sha256": hashlib.sha256(chunks_payload).hexdigest(),
        "stats_sha256": hashlib.sha256(stats_payload).hexdigest(),
    }
    sqlite_payload = metadata_sqlite_bytes(documents, chunks, base_info)
    build_info = {
        **base_info,
        "sqlite_logical_sha256": logical_sqlite_digest_bytes(sqlite_payload),
    }
    return (
        {
            "build-info.json": stable_json_bytes(build_info),
            "chunks.jsonl": chunks_payload,
            "metadata.sqlite3": sqlite_payload,
            "stats.json": stats_payload,
        },
        len(documents),
        len(chunks),
    )


def _verify_existing(output_root: Path, expected: dict[str, bytes]) -> None:
    try:
        validate_directory(output_root)
        children = list(output_root.iterdir())
    except (OSError, FileSafetyError) as error:
        raise PipelineConflictError(f"unsafe output root: {output_root}") from error
    if {path.name for path in children} != _OUTPUT_FILES:
        raise PipelineConflictError(f"output file set mismatch: {output_root}")
    for child in children:
        try:
            validate_regular_file(child)
        except FileSafetyError as error:
            raise PipelineConflictError(f"unsafe output file: {child}") from error
        if child.name == "metadata.sqlite3":
            actual_digest = logical_sqlite_digest(child)
            expected_digest = logical_sqlite_digest_bytes(expected[child.name])
            if actual_digest != expected_digest:
                raise PipelineConflictError("metadata.sqlite3 logical digest mismatch")
        elif child.read_bytes() != expected[child.name]:
            raise PipelineConflictError(f"output differs: {child.name}")


def build_corpus(
    raw_root: Path,
    output_root: Path,
    *,
    config_path: Path = Path("configs/chunking/s2-chunk-v1.json"),
    tokenizer_manifest_path: Path = Path("data/tokenizers/manifest.json"),
    snapshot_manifest_path: Path = Path("data/manifests/snapshot.jsonl"),
) -> BuildResult:
    expected, document_count, chunk_count = _expected_artifacts(
        raw_root,
        config_path=config_path,
        tokenizer_manifest_path=tokenizer_manifest_path,
        snapshot_manifest_path=snapshot_manifest_path,
    )
    parent = output_root.parent
    try:
        ensure_directory(parent)
        with exclusive_lock(parent, output_root.name + ".build"):
            if os.path.lexists(output_root):
                _verify_existing(output_root, expected)
                return BuildResult("skipped", document_count, chunk_count)
            stage, identity = create_stage(parent, f".{output_root.name}.stage-")
            try:
                for name, payload in sorted(expected.items()):
                    write_stage_file(stage, name, payload)
                verify_stage(stage, identity, expected)
                validate_directory(parent)
                atomic_rename_directory_noreplace(stage, output_root)
            except Exception as error:
                raise PipelineError(
                    f"staging retained for manual inspection: {stage}: {error}"
                ) from error
    except FileConflictError as error:
        raise PipelineConflictError(str(error)) from error
    except FileSafetyError as error:
        raise PipelineError(str(error)) from error
    return BuildResult("built", document_count, chunk_count)


def verify_corpus(
    raw_root: Path,
    output_root: Path,
    *,
    config_path: Path = Path("configs/chunking/s2-chunk-v1.json"),
    tokenizer_manifest_path: Path = Path("data/tokenizers/manifest.json"),
    snapshot_manifest_path: Path = Path("data/manifests/snapshot.jsonl"),
) -> BuildResult:
    expected, document_count, chunk_count = _expected_artifacts(
        raw_root,
        config_path=config_path,
        tokenizer_manifest_path=tokenizer_manifest_path,
        snapshot_manifest_path=snapshot_manifest_path,
    )
    _verify_existing(output_root, expected)
    return BuildResult("verified", document_count, chunk_count)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/s2-chunk-v1")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/chunking/s2-chunk-v1.json"))
    parser.add_argument(
        "--tokenizer-manifest",
        type=Path,
        default=Path("data/tokenizers/manifest.json"),
    )
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        default=Path("data/manifests/snapshot.jsonl"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        _common_arguments(subparsers.add_parser(command))
    stats = subparsers.add_parser("stats")
    stats.add_argument(
        "--output-root", type=Path, default=Path("data/processed/s2-chunk-v1")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "stats":
            path = args.output_root / "stats.json"
            validate_regular_file(path)
            print(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False))
            return 0
        function = build_corpus if args.command == "build" else verify_corpus
        function(
            args.raw_root,
            args.output_root,
            config_path=args.config,
            tokenizer_manifest_path=args.tokenizer_manifest,
            snapshot_manifest_path=args.snapshot_manifest,
        )
        return 0
    except (PipelineError, FileSafetyError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
