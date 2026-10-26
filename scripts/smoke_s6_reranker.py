"""Run a deterministic real-model smoke for the frozen S6 reranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from docguard.fs_safety import validate_regular_file
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.reranker import CrossEncoderReranker
from docguard.retrieval.reranker_config import load_reranker_config

QUERY = "What does wal_level control?"
PASSAGES = (
    "wal_level determines how much information is written to the write-ahead log.",
    "CREATE INDEX defines a new index.",
    "unrelated " * 700,
)


def _sha256(path: Path) -> str:
    validate_regular_file(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_smoke(
    model_root: Path,
    manifest_path: Path,
    config_path: Path,
    tokenizer_path: Path,
) -> dict[str, object]:
    config = load_reranker_config(config_path)
    if _sha256(manifest_path) != config.model_manifest_sha256:
        raise ValueError("reranker manifest hash differs from the frozen config")
    if _sha256(tokenizer_path) != config.tokenizer_sha256:
        raise ValueError("reranker tokenizer hash differs from the frozen config")
    assets = verify_model_assets(
        model_root,
        manifest_path,
        expected_model_name=config.model_name,
        expected_revision=config.model_revision,
    )
    reranker = CrossEncoderReranker(assets, tokenizer_path, config)
    try:
        scores = reranker.score_pairs(QUERY, list(PASSAGES))
    finally:
        reranker.close()
    return {
        "schema_version": 1,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "query": QUERY,
        "scores": [
            {"logit": item.logit, "truncated": item.truncated} for item in scores
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/s6-reranker-model.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrieval/s6-reranker-v1.json"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "data/tokenizers/9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e/tokenizer.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = run_smoke(
        arguments.model_root,
        arguments.manifest,
        arguments.config,
        arguments.tokenizer,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
