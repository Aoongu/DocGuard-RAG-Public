"""Direct URL knowledge-base CLI. Generation is an explicit opt-in."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from docguard.discovery.extract import ExtractionError
from docguard.discovery.fetch import FetchError
from docguard.fs_safety import FileSafetyError
from docguard.knowledge.pipeline import KnowledgeBase
from docguard.processing.config import load_chunking_config
from docguard.processing.token_budget import TokenBudget


async def _run(args):
    repo = args.project_root.resolve()
    budget = TokenBudget.from_manifest(repo / "data/tokenizers/manifest.json", repo)
    config = load_chunking_config(repo / "configs/chunking/s2-chunk-v1.json")
    embedder, embedding_id = None, None
    if args.model_root:
        from docguard.retrieval.config import load_dense_config
        from docguard.retrieval.e5_embedder import E5Embedder
        from docguard.retrieval.model_assets import verify_model_assets

        dense_config = load_dense_config(repo / "configs/retrieval/s3-dense-v1.json")
        assets = verify_model_assets(
            args.model_root, repo / "data/manifests/s3-embedding-model.json"
        )
        tokenizer = (
            repo
            / "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json"
        )
        embedder = E5Embedder(assets, tokenizer, dense_config)
        embedding_id = dense_config.model.name + "@" + dense_config.model.revision
    base = KnowledgeBase(
        args.root, budget, config, embedder=embedder, embedding_id=embedding_id
    )
    if args.command == "import":
        return await base.import_urls(
            args.kb, args.url, product=args.product, version=args.version
        )
    if args.command == "info":
        return base.info(args.kb)
    generator = None
    try:
        if args.generate:
            from docguard.generation.client import DeepSeekResponsesClient
            from docguard.generation.config import load_generation_config

            key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not key:
                raise ValueError("Generation API key unavailable")
            generation_config = load_generation_config(
                repo / "configs/generation/s7-deepseek-flash-v1.json"
            )
            generator = DeepSeekResponsesClient(generation_config, api_key=key)
        return await base.answer(
            args.kb, args.question, method=args.method, generator=generator
        )
    finally:
        if generator is not None:
            await generator.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".knowledge"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--model-root", type=Path, help="Verified S3 E5 assets; enables Dense indexing"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser(
        "import", help="Import the explicitly selected URLs as a complete snapshot"
    )
    importer.add_argument("--kb", required=True)
    importer.add_argument("--product", required=True)
    importer.add_argument("--version")
    importer.add_argument("--url", action="append", required=True)
    info = commands.add_parser("info")
    info.add_argument("--kb", required=True)
    query = commands.add_parser("query")
    query.add_argument("--kb", required=True)
    query.add_argument("--question", required=True)
    query.add_argument("--method", choices=("bm25", "hybrid"), default="bm25")
    query.add_argument(
        "--generate",
        action="store_true",
        help="Call the configured DeepSeek API; incurs usage",
    )
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except (
        ValueError,
        OSError,
        FileSafetyError,
        FetchError,
        ExtractionError,
        RuntimeError,
    ) as error:
        # No provider error bodies, user file paths or credentials in CLI failures.
        print(
            json.dumps({"error": "knowledge_error", "type": type(error).__name__}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
