"""Command-line entry point for validating and running S4 evaluations."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from docguard.evaluation.config import load_evaluation_config
from docguard.evaluation.dataset import (
    build_source_index,
    load_questions,
    sha256_file,
    validate_resolved_sources,
    validate_split_collection,
)
from docguard.evaluation.final_protocol import S12TestAuthorization
from docguard.evaluation.runner import TestSplitLockedError, run_evaluation
from docguard.retrieval.bm25_index import BM25Retriever
from docguard.retrieval.config import load_dense_config
from docguard.retrieval.hybrid import HybridRetriever
from docguard.retrieval.hybrid_config import load_hybrid_config
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.reranker import CrossEncoderReranker
from docguard.retrieval.reranker_config import load_reranker_config
from docguard.retrieval.reranking_pipeline import RerankingRetriever
from docguard.retrieval.service import create_default_retriever


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", type=Path, action="append", required=True)
    validate.add_argument("--chunks", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--method", choices=("b0", "b1", "m1", "m2"), default="b1")
    run.add_argument(
        "--eval-config", type=Path, default=Path("configs/evaluation/s4-eval-v1.json")
    )
    run.add_argument(
        "--dense-config", type=Path, default=Path("configs/retrieval/s3-dense-v1.json")
    )
    run.add_argument(
        "--s5-config", type=Path, default=Path("configs/retrieval/s5-hybrid-v1.json")
    )
    run.add_argument(
        "--s6-config", type=Path, default=Path("configs/retrieval/s6-reranker-v1.json")
    )
    run.add_argument("--index-root", type=Path, default=Path(".qdrant/s3-dense-v1"))
    run.add_argument("--bm25-index-root", type=Path, default=Path(".bm25/s5-bm25-v1"))
    run.add_argument("--dense-depth", type=int, choices=(20, 50), default=50)
    run.add_argument("--bm25-depth", type=int, choices=(20, 50), default=50)
    run.add_argument("--rrf-k", type=int, choices=(20, 60), default=60)
    run.add_argument("--model-root", type=Path, default=Path("data/models/s3-e5-small"))
    run.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("data/manifests/s3-embedding-model.json"),
    )
    run.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json"
        ),
    )
    run.add_argument("--rerank-depth", type=int, choices=(10, 20, 50), default=20)
    run.add_argument(
        "--rerank-enabled", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--reranker-model-root",
        type=Path,
        default=Path("data/models/s6-bge-reranker-base"),
    )
    run.add_argument(
        "--reranker-model-manifest",
        type=Path,
        default=Path("data/manifests/s6-reranker-model.json"),
    )
    run.add_argument(
        "--reranker-tokenizer",
        type=Path,
        default=Path(
            "data/tokenizers/9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e/tokenizer.json"
        ),
    )
    return parser


async def _run(
    args: argparse.Namespace,
    *,
    test_authorization: S12TestAuthorization | None = None,
) -> dict[str, object]:
    questions = load_questions(args.dataset)
    validate_split_collection(questions)
    splits = {question.split for question in questions}
    if len(splits) != 1:
        raise ValueError("one run may contain exactly one split")
    split = next(iter(splits))
    if split == "test" and test_authorization is None:
        raise TestSplitLockedError("Test split is frozen and cannot run before S12")
    config = load_evaluation_config(args.eval_config)
    dense_config = load_dense_config(args.dense_config)
    hybrid_config = load_hybrid_config(args.s5_config)
    if hybrid_config.chunks_sha256 != config.chunks_sha256:
        raise ValueError("S5 and evaluation configs have different chunks hashes")
    reranker_config = (
        load_reranker_config(args.s6_config) if args.method == "m2" else None
    )
    if reranker_config is not None:
        if reranker_config.chunks_sha256 != config.chunks_sha256:
            raise ValueError("S6 and evaluation configs have different chunks hashes")
        if reranker_config.s5_config_sha256 != sha256_file(args.s5_config):
            raise ValueError("S6 config does not bind the supplied S5 config")
        if reranker_config.model_manifest_sha256 != sha256_file(
            args.reranker_model_manifest
        ):
            raise ValueError("S6 model manifest hash differs from the frozen config")
        if reranker_config.tokenizer_sha256 != sha256_file(args.reranker_tokenizer):
            raise ValueError("S6 tokenizer hash differs from the frozen config")
    dense_enabled = args.method in {"b1", "m1", "m2"}
    bm25_enabled = args.method in {"b0", "m1", "m2"}
    dense_manifest_sha256 = (
        sha256_file(args.index_root / "build-info.json") if dense_enabled else None
    )
    bm25_manifest_sha256 = (
        sha256_file(args.bm25_index_root / "manifest.json") if bm25_enabled else None
    )
    dense_depth = reranker_config.dense_depth if reranker_config else args.dense_depth
    bm25_depth = reranker_config.bm25_depth if reranker_config else args.bm25_depth
    rrf_k = reranker_config.rrf_k if reranker_config else args.rrf_k
    metadata: dict[str, object] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "git_commit": _git_commit(),
        "split": split,
        "dataset_version": config.dataset_version,
        "dataset_sha256": sha256_file(args.dataset),
        "config_sha256": sha256_file(args.eval_config),
        "corpus_manifest_sha256": config.corpus_manifest_sha256,
        "chunks_sha256": config.chunks_sha256,
        "chunking_version": config.chunking_version,
        "method": args.method,
        "s5_config_sha256": sha256_file(args.s5_config),
        "dense_index_manifest_sha256": dense_manifest_sha256,
        "bm25_index_manifest_sha256": bm25_manifest_sha256,
        "model_name": dense_config.model.name if dense_enabled else None,
        "model_revision": dense_config.model.revision if dense_enabled else None,
        "parameters": {
            "top_k": config.retrieval_top_k,
            "dense_enabled": dense_enabled,
            "bm25_enabled": bm25_enabled,
            "dense_depth": dense_depth,
            "bm25_depth": bm25_depth,
            "rrf_k": rrf_k,
        },
    }
    hybrid = HybridRetriever(
        lambda: create_default_retriever(
            index_root=args.index_root,
            config_path=args.dense_config,
            model_root=args.model_root,
            model_manifest_path=args.model_manifest,
            tokenizer_path=args.tokenizer,
        ),
        lambda: BM25Retriever(
            args.bm25_index_root,
            config_path=args.s5_config,
        ),
    )
    retriever: HybridRetriever | RerankingRetriever = hybrid
    if reranker_config is not None:
        verified_assets = verify_model_assets(
            args.reranker_model_root,
            args.reranker_model_manifest,
            expected_model_name=reranker_config.model_name,
            expected_revision=reranker_config.model_revision,
        )
        parameters = metadata["parameters"]
        assert isinstance(parameters, dict)
        parameters.update(
            {
                "rerank_enabled": args.rerank_enabled,
                "rerank_depth": args.rerank_depth,
            }
        )
        metadata.update(
            {
                "s6_config_sha256": sha256_file(args.s6_config),
                "reranker_model_name": reranker_config.model_name,
                "reranker_model_revision": reranker_config.model_revision,
                "reranker_model_manifest_sha256": sha256_file(
                    args.reranker_model_manifest
                ),
                "reranker_tokenizer_sha256": sha256_file(args.reranker_tokenizer),
                "reranker_max_length": reranker_config.max_length,
            }
        )
        retriever = RerankingRetriever(
            hybrid,
            lambda: CrossEncoderReranker(
                verified_assets,
                args.reranker_tokenizer,
                reranker_config,
            ),
            reranker_config,
        )
    try:
        return await run_evaluation(
            questions,
            args.output,
            retriever,
            config,
            metadata,
            test_authorization=test_authorization,
        )
    finally:
        await retriever.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            questions = []
            for path in args.dataset:
                questions.extend(load_questions(path))
            validate_split_collection(questions)
            validate_resolved_sources(questions, build_source_index(args.chunks))
            result: dict[str, object] = {
                "status": "valid",
                "question_count": len(questions),
                "dataset_sha256": {
                    str(path): sha256_file(path) for path in args.dataset
                },
            }
        else:
            result = asyncio.run(_run(args))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
