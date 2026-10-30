"""Lazy, shared copies of the same verified E5 and BGE models as PostgreSQL."""

import threading
from pathlib import Path

from docguard.retrieval.config import load_dense_config
from docguard.retrieval.e5_embedder import E5Embedder
from docguard.retrieval.model_assets import verify_model_assets
from docguard.retrieval.reranker import CrossEncoderReranker
from docguard.retrieval.reranker_config import load_reranker_config


class LocalRetrievalModels:
    def __init__(self, project_root: Path, asset_root: Path):
        self.project_root, self.asset_root = project_root, asset_root
        self.dense_config = load_dense_config(
            project_root / "configs/retrieval/s3-dense-v1.json"
        )
        self.reranker_config = load_reranker_config(
            project_root / "configs/retrieval/s6-reranker-v1.json"
        )
        self.embedding_id = (
            self.dense_config.model.name + "@" + self.dense_config.model.revision
        )
        self.reranker_id = (
            self.reranker_config.model_name + "@" + self.reranker_config.model_revision
        )
        self._embedding = self._reranking = None
        self._lock = threading.RLock()

    def _embedder(self):
        if self._embedding is None:
            assets = verify_model_assets(
                self.asset_root / "data/models/s3-e5-small",
                self.project_root / "data/manifests/s3-embedding-model.json",
                expected_model_name=self.dense_config.model.name,
                expected_revision=self.dense_config.model.revision,
            )
            tokenizer = (
                self.project_root
                / "data/tokenizers/0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39/tokenizer.json"
            )
            self._embedding = E5Embedder(assets, tokenizer, self.dense_config)
        return self._embedding

    async def embed_passages(self, texts):
        with self._lock:
            return await self._embedder().embed_passages(texts)

    async def embed_queries(self, texts):
        with self._lock:
            return await self._embedder().embed_queries(texts)

    def score_pairs(self, query, passages):
        with self._lock:
            if self._reranking is None:
                config = self.reranker_config
                assets = verify_model_assets(
                    self.asset_root / "data/models/s6-bge-reranker-base",
                    self.project_root / config.model_manifest_path,
                    expected_model_name=config.model_name,
                    expected_revision=config.model_revision,
                )
                self._reranking = CrossEncoderReranker(
                    assets, self.project_root / config.tokenizer_path, config
                )
            return self._reranking.score_pairs(query, passages)
