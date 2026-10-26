"""Fixed command-line entry point for the sealed one-shot S12 Test sequence."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from docguard.api.runtime import build_live_m4_pipeline
from docguard.evaluation.config import load_evaluation_config
from docguard.evaluation.dataset import load_questions, sha256_file
from docguard.evaluation.final_protocol import (
    S12FinalProtocol,
    authorize_test_run,
    load_final_protocol,
    verify_final_protocol_inputs,
)
from docguard.evaluation.final_runner import run_final_evaluation
from docguard.evaluation.generation_cli import _run_b2
from docguard.evaluation.runner import run_evaluation
from docguard.evaluation.s12_execution import (
    begin_method,
    complete_method,
    load_ledger,
    save_ledger,
    validate_runtime_status,
    validate_seal,
)
from docguard.retrieval.bm25_index import BM25Retriever
from docguard.retrieval.config import load_dense_config
from docguard.retrieval.hybrid import HybridRetriever
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.reranker import CrossEncoderReranker
from docguard.retrieval.reranker_config import load_reranker_config
from docguard.retrieval.reranking_pipeline import RerankingRetriever
from docguard.retrieval.service import create_default_retriever

_RETRIEVAL_METHODS = {"b0", "b1", "m1", "m2"}
_FINAL_METHODS = {"m3", "m4", "m4_no_version_filter", "m4_no_citation_validation"}


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_commit(root: Path) -> str:
    return _git_output(root, "rev-parse", "HEAD")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--method",
        choices=(
            "b0",
            "b1",
            "m1",
            "m2",
            "b2",
            "m3",
            "m4",
            "m4_no_version_filter",
            "m4_no_citation_validation",
        ),
        required=True,
    )
    run.add_argument("--project-root", type=Path, default=Path.cwd())
    run.add_argument("--asset-root", type=Path, default=Path.cwd())
    return parser


def _metadata(
    root: Path,
    protocol: S12FinalProtocol,
    method: str,
    *,
    git_commit: str,
) -> dict[str, object]:
    frozen = protocol.methods[method]
    return {
        "schema_version": 1,
        "protocol_version": protocol.protocol_version,
        "run_id": frozen.run_id,
        "git_commit": git_commit,
        "split": "test",
        "dataset_version": protocol.dataset.version,
        "dataset_sha256": protocol.dataset.test_sha256,
        "protocol_config_sha256": sha256_file(
            root / "configs/evaluation/s12-final-v1.json"
        ),
        "method": method,
        "parameters": dict(frozen.parameters),
        "max_retries": protocol.max_retries,
        "existing_record_policy": protocol.existing_record_policy,
    }


async def _run_retrieval(
    root: Path,
    asset_root: Path,
    protocol: S12FinalProtocol,
    method: str,
) -> dict[str, object]:
    authorization = authorize_test_run(root, protocol, method)
    questions = load_questions(root / protocol.dataset.test_path)
    config_path = root / "configs/evaluation/s4-eval-v1.json"
    dense_config_path = root / "configs/retrieval/s3-dense-v1.json"
    s5_path = root / "configs/retrieval/s5-hybrid-v1.json"
    s6_path = root / "configs/retrieval/s6-reranker-v1.json"
    config = load_evaluation_config(config_path)
    dense_config = load_dense_config(dense_config_path)
    parameters = dict(protocol.methods[method].parameters)
    dense_enabled = method in {"b1", "m1", "m2"}
    bm25_enabled = method in {"b0", "m1", "m2"}
    dense_index = asset_root / ".qdrant/s3-dense-v1"
    bm25_index = asset_root / ".bm25/s5-bm25-v1"
    metadata: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": protocol.protocol_version,
        "run_id": protocol.methods[method].run_id,
        "git_commit": _git_commit(root),
        "split": "test",
        "dataset_version": protocol.dataset.version,
        "dataset_sha256": protocol.dataset.test_sha256,
        "config_sha256": sha256_file(config_path),
        "corpus_manifest_sha256": config.corpus_manifest_sha256,
        "chunks_sha256": config.chunks_sha256,
        "chunking_version": config.chunking_version,
        "method": method,
        "s5_config_sha256": sha256_file(s5_path),
        "dense_index_manifest_sha256": (
            sha256_file(dense_index / "build-info.json") if dense_enabled else None
        ),
        "bm25_index_manifest_sha256": (
            sha256_file(bm25_index / "manifest.json") if bm25_enabled else None
        ),
        "model_name": dense_config.model.name if dense_enabled else None,
        "model_revision": dense_config.model.revision if dense_enabled else None,
        "parameters": parameters,
    }
    hybrid = HybridRetriever(
        lambda: create_default_retriever(
            index_root=dense_index,
            config_path=dense_config_path,
            model_root=asset_root / "data/models/s3-e5-small",
            model_manifest_path=root / "data/manifests/s3-embedding-model.json",
            tokenizer_path=root
            / "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json",
        ),
        lambda: BM25Retriever(bm25_index, config_path=s5_path),
    )
    retriever: HybridRetriever | RerankingRetriever = hybrid
    if method == "m2":
        reranker_config = load_reranker_config(s6_path)
        reranker_manifest = root / reranker_config.model_manifest_path
        assets = verify_model_assets(
            asset_root / "data/models/s6-bge-reranker-base",
            reranker_manifest,
            expected_model_name=reranker_config.model_name,
            expected_revision=reranker_config.model_revision,
        )
        metadata.update(
            {
                "s6_config_sha256": sha256_file(s6_path),
                "reranker_model_name": reranker_config.model_name,
                "reranker_model_revision": reranker_config.model_revision,
                "reranker_model_manifest_sha256": sha256_file(reranker_manifest),
                "reranker_tokenizer_sha256": sha256_file(
                    root / reranker_config.tokenizer_path
                ),
                "reranker_max_length": reranker_config.max_length,
            }
        )
        retriever = RerankingRetriever(
            hybrid,
            lambda: CrossEncoderReranker(
                assets, root / reranker_config.tokenizer_path, reranker_config
            ),
            reranker_config,
        )
    try:
        return await run_evaluation(
            questions,
            authorization.output_dir,
            retriever,
            config,
            metadata,
            test_authorization=authorization,
        )
    finally:
        await retriever.close()


async def _run_b2_fixed(
    root: Path, asset_root: Path, protocol: S12FinalProtocol
) -> dict[str, object]:
    authorization = authorize_test_run(root, protocol, "b2")
    args = argparse.Namespace(
        command="run-b2",
        dataset=root / protocol.dataset.test_path,
        output=authorization.output_dir,
        run_id=protocol.methods["b2"].run_id,
        generation_config=root / "configs/generation/s7-deepseek-flash-v1.json",
        mock_response_file=None,
        index_root=asset_root / ".qdrant/s3-dense-v1",
        dense_config=root / "configs/retrieval/s3-dense-v1.json",
        model_root=asset_root / "data/models/s3-e5-small",
        model_manifest=root / "data/manifests/s3-embedding-model.json",
        tokenizer=root
        / "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json",
    )
    return await _run_b2(args, test_authorization=authorization)


async def _run_final(
    root: Path, asset_root: Path, protocol: S12FinalProtocol, method: str
) -> dict[str, object]:
    authorization = authorize_test_run(root, protocol, method)
    questions = load_questions(root / protocol.dataset.test_path)
    metadata = _metadata(root, protocol, method, git_commit=_git_commit(root))
    pipeline = build_live_m4_pipeline(root, asset_root)
    try:
        return await run_final_evaluation(
            questions,
            authorization.output_dir,
            pipeline,
            metadata,
            test_authorization=authorization,
        )
    finally:
        await pipeline.close()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    root = args.project_root.resolve()
    asset_root = args.asset_root.resolve()
    protocol_path = root / "configs/evaluation/s12-final-v1.json"
    protocol = load_final_protocol(protocol_path)
    verify_final_protocol_inputs(root, protocol)
    commit = _git_commit(root)
    seal = json.loads(
        (root / "reports/s12_final/preflight.json").read_text(encoding="utf-8")
    )
    validate_seal(
        seal,
        commit,
        sha256_file(protocol_path),
        protocol.dataset.test_sha256,
        protocol.method_order,
    )
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    validate_runtime_status(status)
    ledger_path = root / "reports/s12_final/execution-ledger.json"
    ledger = begin_method(
        load_ledger(ledger_path), args.method, protocol.method_order, commit
    )
    save_ledger(ledger_path, ledger)
    if args.method in _RETRIEVAL_METHODS:
        summary = await _run_retrieval(root, asset_root, protocol, args.method)
    elif args.method == "b2":
        summary = await _run_b2_fixed(root, asset_root, protocol)
    elif args.method in _FINAL_METHODS:
        summary = await _run_final(root, asset_root, protocol, args.method)
    else:  # pragma: no cover - argparse and protocol both reject this
        raise ValueError("unsupported S12 method")
    output_dir = root / protocol.output_root / protocol.methods[args.method].output_dir
    ledger = complete_method(ledger, args.method, output_dir)
    save_ledger(ledger_path, ledger)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(_run(args))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
