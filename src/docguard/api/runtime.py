"""Lazy production wiring for the frozen M4 pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from docguard.api.models import DecisionResponse, QueryResponse
from docguard.generation.client import DeepSeekResponsesClient
from docguard.generation.config import load_generation_config
from docguard.generation.service import GenerationService
from docguard.refusal.config import load_refusal_config
from docguard.refusal.pipeline import M4Pipeline
from docguard.refusal.policy import RefusalThresholds
from docguard.retrieval.bm25_index import BM25Retriever
from docguard.retrieval.hybrid import HybridRetriever
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.reranker import CrossEncoderReranker
from docguard.retrieval.reranker_config import load_reranker_config
from docguard.retrieval.reranking_pipeline import RerankingRetriever
from docguard.retrieval.service import create_default_retriever
from docguard.service.query import QueryOutcome, QueryService


def _frozen_thresholds(path: Path) -> RefusalThresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected") if isinstance(payload, dict) else None
    values = selected.get("thresholds") if isinstance(selected, dict) else None
    if not isinstance(values, dict) or set(values) != {
        "top1",
        "top3_mean",
        "coverage",
        "margin",
    }:
        raise ValueError("S8 selection does not contain frozen thresholds")
    return RefusalThresholds(
        top1=float(values["top1"]),
        top3_mean=float(values["top3_mean"]),
        coverage=float(values["coverage"]),
        margin=float(values["margin"]),
    )


def build_live_m4_pipeline(project_root: Path, asset_root: Path) -> M4Pipeline:
    """Build the frozen production M4 pipeline from verified local assets."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DeepSeek credential is unavailable")
    dense_config = project_root / "configs/retrieval/s3-dense-v1.json"
    dense_tokenizer = project_root / (
        "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/"
        "tokenizer.json"
    )
    hybrid_config = project_root / "configs/retrieval/s5-hybrid-v1.json"
    reranker_config_path = project_root / "configs/retrieval/s6-reranker-v1.json"
    reranker_config = load_reranker_config(reranker_config_path)
    reranker_tokenizer = project_root / reranker_config.tokenizer_path
    reranker_model_root = asset_root / "data/models/s6-bge-reranker-base"
    reranker_manifest = project_root / reranker_config.model_manifest_path
    reranker_assets = verify_model_assets(
        reranker_model_root,
        reranker_manifest,
        expected_model_name=reranker_config.model_name,
        expected_revision=reranker_config.model_revision,
    )
    hybrid = HybridRetriever(
        lambda: create_default_retriever(
            index_root=asset_root / ".qdrant/s3-dense-v1",
            config_path=dense_config,
            model_root=asset_root / "data/models/s3-e5-small",
            model_manifest_path=project_root / "data/manifests/s3-embedding-model.json",
            tokenizer_path=dense_tokenizer,
            qdrant_url=os.environ.get("DOCGUARD_QDRANT_URL") or None,
        ),
        lambda: BM25Retriever(
            asset_root / ".bm25/s5-bm25-v1", config_path=hybrid_config
        ),
    )
    reranked = RerankingRetriever(
        hybrid,
        lambda: CrossEncoderReranker(
            reranker_assets, reranker_tokenizer, reranker_config
        ),
        reranker_config,
    )
    generation_config = load_generation_config(
        project_root / "configs/generation/s7-deepseek-flash-v1.json"
    )
    generation = GenerationService(
        generation_config,
        lambda: DeepSeekResponsesClient(generation_config, api_key=api_key),
    )
    refusal_config = load_refusal_config(
        project_root / "configs/refusal/s8-refusal-v1.json"
    )
    thresholds = _frozen_thresholds(
        project_root / "reports/s8_refusal/validation_selected/selection.json"
    )
    return M4Pipeline(reranked, generation, refusal_config, thresholds)


def build_live_query_service(
    project_root: Path, asset_root: Path, timeout_seconds: float
) -> QueryService:
    return QueryService(
        build_live_m4_pipeline(project_root, asset_root), timeout_seconds
    )


def _unavailable(
    question: str, requested_version: str | None, request_id: str
) -> QueryOutcome:
    response = QueryResponse(
        request_id=request_id,
        question=question,
        requested_version=requested_version,
        route="m4",
        decision=DecisionResponse(
            action="error",
            reason_code="UPSTREAM_ERROR",
            confidence=1.0,
            triggered_rule="runtime_dependency_unavailable",
        ),
        answer=None,
        citations=[],
        evidence=[],
        latency_ms={"end_to_end": 0.0},
        model_info=None,
    )
    return QueryOutcome(response, 503)


class LazyQueryService:
    def __init__(self, factory: Callable[[], QueryService]) -> None:
        self._factory = factory
        self._service: QueryService | None = None

    async def query(
        self, question: str, requested_version: str | None, *, request_id: str
    ) -> QueryOutcome:
        if self._service is None:
            try:
                self._service = self._factory()
            except (OSError, RuntimeError, TypeError, ValueError):
                return _unavailable(question, requested_version, request_id)
        try:
            return await self._service.query(
                question, requested_version, request_id=request_id
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return _unavailable(question, requested_version, request_id)

    async def close(self) -> None:
        if self._service is not None:
            await self._service.close()
            self._service = None


def dependency_probe(
    project_root: Path, asset_root: Path
) -> dict[str, tuple[bool, str | None]]:
    paths = {
        "corpus_metadata": asset_root / "data/processed/s2-chunk-v1/metadata.sqlite3",
        "dense_index": asset_root / ".qdrant/s3-dense-v1/qdrant",
        "bm25_index": asset_root / ".bm25/s5-bm25-v1/bm25.sqlite3",
        "embedding_model": asset_root / "data/models/s3-e5-small/onnx/model.onnx",
        "reranker_model": asset_root
        / "data/models/s6-bge-reranker-base/onnx/model.onnx",
        "api_config": project_root / "configs/api/s9-api-v1.json",
    }
    result = {
        name: (path.exists(), None if path.exists() else "unavailable")
        for name, path in paths.items()
    }
    result["generation_api"] = (
        bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        None if os.environ.get("DEEPSEEK_API_KEY", "").strip() else "unavailable",
    )
    return result


async def unavailable_evaluation_executor(
    run_id: str, method: str, split: str
) -> dict[str, object]:
    del run_id, method, split
    raise RuntimeError("online evaluation execution is not configured")
