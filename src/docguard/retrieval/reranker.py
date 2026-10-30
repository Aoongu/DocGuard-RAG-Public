"""Deterministic offline Cross-Encoder scoring for S6."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

from docguard.fs_safety import validate_regular_file
from docguard.retrieval.hybrid import HybridSearchHit
from docguard.retrieval.model_assets import VerifiedModelAssets
from docguard.retrieval.reranker_config import RerankerConfig


class RerankerError(RuntimeError):
    """Raised when reranking violates the frozen S6 contract."""


@dataclass(frozen=True, slots=True)
class PairScore:
    logit: float
    truncated: bool


@dataclass(frozen=True, slots=True)
class RerankedSearchHit:
    rank: int
    chunk_id: str
    score: float
    rerank_score: float
    rerank_truncated: bool
    original_rrf_rank: int
    rrf_score: float | None
    sources: tuple[str, ...]
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    postgresql_version: str
    topic: str
    section_path: tuple[str, ...]
    canonical_url: str
    anchor: str
    text: str


class PairScorer(Protocol):
    def score_pairs(self, query: str, passages: list[str]) -> list[PairScore]: ...


class _Encoding(Protocol):
    ids: list[int]
    attention_mask: list[int]
    overflowing: list[object]


class _Tokenizer(Protocol):
    def token_to_id(self, token: str) -> int | None: ...

    def enable_truncation(self, **kwargs: object) -> None: ...

    def enable_padding(self, **kwargs: object) -> None: ...

    def encode_batch(self, pairs: list[tuple[str, str]]) -> list[_Encoding]: ...


class _SessionInput(Protocol):
    name: str


class _Session(Protocol):
    def get_inputs(self) -> list[_SessionInput]: ...

    def run(
        self, output_names: None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


TokenizerFactory = Callable[[str], _Tokenizer]
SessionFactory = Callable[..., _Session]


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip() or "\0" in query:
        raise ValueError("query must be a non-empty string without NUL")


def score_ranked_passages(
    query: str,
    passages: Sequence[tuple[str, int, str]],
    scorer: PairScorer,
    *,
    top_k: int,
) -> list[tuple[float, int, str, PairScore]]:
    """Score and stably order passages without assuming a document product."""
    _validate_query(query)
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not passages:
        return []
    scores = scorer.score_pairs(query, [text for _, _, text in passages])
    if len(scores) != len(passages):
        raise RerankerError("reranker score count differs from candidate count")
    ranked = []
    seen = set()
    for (key, rank, text), pair in zip(passages, scores, strict=True):
        if key in seen or type(rank) is not int or rank <= 0 or not text.strip():
            raise RerankerError("Invalid or duplicate reranker candidate")
        seen.add(key)
        logit = float(pair.logit)
        if not math.isfinite(logit):
            raise RerankerError("reranker score must be finite")
        if type(pair.truncated) is not bool:
            raise RerankerError("reranker truncation flag must be boolean")
        ranked.append((logit, rank, key, pair))
    return sorted(ranked, key=lambda item: (-item[0], item[1], item[2]))[:top_k]


def rerank_hits(
    query: str,
    candidates: Sequence[HybridSearchHit],
    scorer: PairScorer,
    *,
    top_k: int,
) -> list[RerankedSearchHit]:
    """Score M1 candidates and return a stable provenance-preserving order."""
    _validate_query(query)
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.chunk_id in seen:
            raise RerankerError(f"duplicate candidate chunk_id: {candidate.chunk_id}")
        seen.add(candidate.chunk_id)
        if not isinstance(candidate.text, str) or not candidate.text.strip():
            raise RerankerError(f"candidate passage is empty: {candidate.chunk_id}")
        if type(candidate.rank) is not int or candidate.rank <= 0:
            raise RerankerError(f"candidate rank is invalid: {candidate.chunk_id}")
    if not candidates:
        return []

    mapping = {candidate.chunk_id: candidate for candidate in candidates}
    scored = score_ranked_passages(
        query, [(c.chunk_id, c.rank, c.text) for c in candidates], scorer, top_k=top_k
    )

    output: list[RerankedSearchHit] = []
    for rank, (logit, original_rank, chunk_id, pair_score) in enumerate(
        scored, start=1
    ):
        candidate = mapping[chunk_id]
        output.append(
            RerankedSearchHit(
                rank=rank,
                chunk_id=candidate.chunk_id,
                score=logit,
                rerank_score=logit,
                rerank_truncated=pair_score.truncated,
                original_rrf_rank=original_rank,
                rrf_score=candidate.rrf_score,
                sources=candidate.sources,
                dense_rank=candidate.dense_rank,
                dense_score=candidate.dense_score,
                bm25_rank=candidate.bm25_rank,
                bm25_score=candidate.bm25_score,
                postgresql_version=candidate.postgresql_version,
                topic=candidate.topic,
                section_path=candidate.section_path,
                canonical_url=candidate.canonical_url,
                anchor=candidate.anchor,
                text=candidate.text,
            )
        )
    return output


class CrossEncoderReranker:
    """Fixed BGE reranker implemented directly with tokenizer and ONNX Runtime."""

    def __init__(
        self,
        model_assets: VerifiedModelAssets,
        tokenizer_path: Path,
        config: RerankerConfig,
        *,
        tokenizer_factory: TokenizerFactory | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        model_path = model_assets.model_root / "onnx" / "model.onnx"
        validate_regular_file(model_path)
        validate_regular_file(tokenizer_path)
        tokenizer_loader = tokenizer_factory or Tokenizer.from_file
        session_loader = session_factory or onnxruntime.InferenceSession
        self._config = config
        self._tokenizer = tokenizer_loader(str(tokenizer_path))
        pad_id = self._tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise RerankerError("tokenizer does not define the <pad> token")
        self._tokenizer.enable_truncation(
            max_length=config.max_length,
            strategy=config.truncation,
        )
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        self._session: _Session | None = session_loader(
            str(model_path), providers=[config.execution_provider]
        )
        self._input_names = {item.name for item in self._session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(self._input_names):
            raise RerankerError("ONNX model inputs do not match the reranker contract")

    def score_pairs(self, query: str, passages: list[str]) -> list[PairScore]:
        _validate_query(query)
        if not isinstance(passages, list):
            raise TypeError("passages must be a list")
        if any(
            not isinstance(passage, str) or not passage.strip() or "\0" in passage
            for passage in passages
        ):
            raise ValueError("passages must contain non-empty strings without NUL")
        if not passages:
            return []
        if self._session is None:
            raise RerankerError("reranker is closed")

        result: list[PairScore] = []
        for offset in range(0, len(passages), self._config.batch_size):
            batch = passages[offset : offset + self._config.batch_size]
            encoded = self._tokenizer.encode_batch(
                [(query, passage) for passage in batch]
            )
            if len(encoded) != len(batch):
                raise RerankerError("tokenizer output count differs from pair count")
            input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            attention_mask = np.asarray(
                [item.attention_mask for item in encoded], dtype=np.int64
            )
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise RerankerError("tokenizer arrays have an unexpected shape")
            feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)
            try:
                outputs = self._session.run(None, feeds)
            except Exception as error:
                raise RerankerError("ONNX reranker inference failed") from error
            if not outputs:
                raise RerankerError("ONNX reranker returned no outputs")
            logits = np.asarray(outputs[0], dtype=np.float32)
            if logits.shape != (len(batch), 1):
                raise RerankerError("ONNX reranker output has an unexpected shape")
            if not np.isfinite(logits).all():
                raise RerankerError("ONNX reranker output contains non-finite values")
            result.extend(
                PairScore(float(logits[index, 0]), bool(encoding.overflowing))
                for index, encoding in enumerate(encoded)
            )
        return result

    def close(self) -> None:
        self._session = None
