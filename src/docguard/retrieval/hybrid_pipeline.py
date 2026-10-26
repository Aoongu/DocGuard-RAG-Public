"""Command-line build, verification, and statistics for the S5 BM25 index."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from docguard.retrieval.bm25_index import (
    BM25Retriever,
    bm25_index_stats,
    build_bm25_index,
    verify_bm25_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docguard-s5-bm25")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--chunks", type=Path, required=True)
        command.add_argument(
            "--config",
            type=Path,
            default=Path("configs/retrieval/s5-hybrid-v1.json"),
        )
        command.add_argument(
            "--output-root" if name == "build" else "--index-root",
            type=Path,
            required=True,
        )
    stats = commands.add_parser("stats")
    stats.add_argument("--index-root", type=Path, required=True)
    search = commands.add_parser("search")
    search.add_argument("--index-root", type=Path, required=True)
    search.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrieval/s5-hybrid-v1.json"),
    )
    search.add_argument("--query", required=True)
    search.add_argument("--version", choices=("17.11", "18.6"))
    search.add_argument("--top-k", type=int, default=5)
    return parser


async def _search(arguments: argparse.Namespace) -> list[dict[str, object]]:
    retriever = BM25Retriever(arguments.index_root, config_path=arguments.config)
    try:
        hits = await retriever.search(
            arguments.query, arguments.version, arguments.top_k, bm25_enabled=True
        )
        return [asdict(hit) for hit in hits]
    finally:
        await retriever.close()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        result = build_bm25_index(
            arguments.chunks, arguments.output_root, config_path=arguments.config
        )
    elif arguments.command == "verify":
        result = verify_bm25_index(
            arguments.index_root, arguments.chunks, config_path=arguments.config
        )
    elif arguments.command == "stats":
        result = bm25_index_stats(arguments.index_root)
    else:
        print(json.dumps(asyncio.run(_search(arguments)), ensure_ascii=False, sort_keys=True))
        return 0
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
