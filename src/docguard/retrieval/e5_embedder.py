"""Offline ONNX implementation of the fixed multilingual E5 embedder."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime
from ragbits.core.embeddings.base import VectorSize
from ragbits.core.embeddings.dense.base import DenseEmbedder
from ragbits.core.options import Options
from tokenizers import Tokenizer

from docguard.fs_safety import validate_regular_file
from docguard.retrieval.config import DenseConfig
from docguard.retrieval.model_assets import VerifiedModelAssets


class E5EmbeddingError(RuntimeError):
    """Raised when local E5 inference violates the frozen contract."""


class _Encoding(Protocol):
    ids: list[int]
    attention_mask: list[int]


class _Tokenizer(Protocol):
    def token_to_id(self, token: str) -> int | None: ...

    def enable_truncation(self, **kwargs: object) -> None: ...

    def enable_padding(self, **kwargs: object) -> None: ...

    def encode_batch(self, texts: list[str]) -> list[_Encoding]: ...


class _SessionInput(Protocol):
    name: str


class _Session(Protocol):
    def get_inputs(self) -> list[_SessionInput]: ...

    def run(
        self, output_names: None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


TokenizerFactory = Callable[[str], _Tokenizer]
SessionFactory = Callable[..., _Session]


class E5Embedder(DenseEmbedder[Options]):
    """Ragbits dense embedder with explicit E5 query/passage semantics."""

    options_cls = Options

    def __init__(
        self,
        model_assets: VerifiedModelAssets,
        tokenizer_path: Path,
        config: DenseConfig,
        *,
        tokenizer_factory: TokenizerFactory | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        super().__init__(default_options=Options())
        model_path = model_assets.model_root / "onnx" / "model.onnx"
        validate_regular_file(model_path)
        validate_regular_file(tokenizer_path)
        self._config = config
        tokenizer_loader = tokenizer_factory or Tokenizer.from_file
        session_loader = session_factory or onnxruntime.InferenceSession
        self._tokenizer = tokenizer_loader(str(tokenizer_path))
        pad_id = self._tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise E5EmbeddingError("tokenizer does not define the <pad> token")
        self._tokenizer.enable_truncation(max_length=config.model.max_length)
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        self._session = session_loader(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = {item.name for item in self._session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(self._input_names):
            raise E5EmbeddingError("ONNX model inputs do not match the E5 contract")

    async def get_vector_size(self) -> VectorSize:
        return VectorSize(size=self._config.model.embedding_dimension, is_sparse=False)

    async def embed_text(
        self, data: list[str], options: Options | None = None
    ) -> list[list[float]]:
        del data, options
        raise TypeError("use embed_passages or embed_queries to preserve E5 semantics")

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self._config.model.passage_prefix)

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self._config.model.query_prefix)

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")
        if any(
            not isinstance(text, str) or not text.strip() or "\0" in text
            for text in texts
        ):
            raise ValueError("texts must contain non-empty strings without NUL")

    def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        self._validate_texts(texts)
        if not texts:
            return []
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self._config.batch_size):
            batch = texts[offset : offset + self._config.batch_size]
            encoded = self._tokenizer.encode_batch([prefix + text for text in batch])
            input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            attention_mask = np.asarray(
                [item.attention_mask for item in encoded], dtype=np.int64
            )
            feeds = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)
            outputs = self._session.run(None, feeds)
            if not outputs:
                raise E5EmbeddingError("ONNX model returned no outputs")
            vectors.extend(self._pool_and_normalize(outputs[0], attention_mask))
        return vectors

    def _pool_and_normalize(
        self, hidden_state: Any, attention_mask: np.ndarray
    ) -> list[list[float]]:
        hidden = np.asarray(hidden_state, dtype=np.float32)
        if hidden.ndim != 3 or hidden.shape[:2] != attention_mask.shape:
            raise E5EmbeddingError("ONNX hidden state has an unexpected shape")
        if hidden.shape[2] != self._config.model.embedding_dimension:
            raise E5EmbeddingError("ONNX embedding dimension does not match config")
        if not np.isfinite(hidden).all():
            raise E5EmbeddingError("ONNX output contains non-finite values")
        mask = attention_mask.astype(np.float32)[:, :, None]
        denominator = mask.sum(axis=1)
        if np.any(denominator <= 0):
            raise E5EmbeddingError("tokenizer produced an empty attention mask")
        pooled = (hidden * mask).sum(axis=1) / denominator
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
            raise E5EmbeddingError("embedding norm is zero or non-finite")
        normalized = pooled / norms
        result = normalized.astype(np.float32).tolist()
        if any(
            len(vector) != self._config.model.embedding_dimension
            or any(not math.isfinite(value) for value in vector)
            for vector in result
        ):
            raise E5EmbeddingError("normalized embedding is invalid")
        return result
