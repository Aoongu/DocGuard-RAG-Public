from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from docguard.retrieval.hybrid import HybridSearchHit
from docguard.retrieval.model_assets import VerifiedModelAssets
from docguard.retrieval.reranker import (
    CrossEncoderReranker,
    PairScore,
    RerankerError,
    rerank_hits,
)
from docguard.retrieval.reranker_config import load_reranker_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_reranker_config(
    PROJECT_ROOT / "configs" / "retrieval" / "s6-reranker-v1.json"
)


def _hit(rank: int, chunk_id: str, text: str = "passage") -> HybridSearchHit:
    return HybridSearchHit(
        rank=rank,
        chunk_id=chunk_id,
        score=1 / (20 + rank),
        rrf_score=1 / (20 + rank),
        sources=("dense", "bm25"),
        dense_rank=rank,
        dense_score=0.8 - rank / 100,
        bm25_rank=rank + 1,
        bm25_score=2.0 - rank / 100,
        postgresql_version="18.6",
        topic="sql_commands",
        section_path=("SELECT",),
        canonical_url="https://www.postgresql.org/docs/18/sql-select.html",
        anchor=f"A-{rank}",
        text=text,
    )


class FixedScorer:
    def __init__(self, scores: list[PairScore]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score_pairs(self, query: str, passages: list[str]) -> list[PairScore]:
        self.calls.append((query, passages))
        return self.scores


def test_rerank_hits_orders_logits_and_preserves_provenance() -> None:
    candidates = [_hit(1, "b"), _hit(2, "a"), _hit(3, "c")]
    scorer = FixedScorer(
        [PairScore(0.4, False), PairScore(0.9, True), PairScore(0.4, False)]
    )

    results = rerank_hits("create index", candidates, scorer, top_k=3)

    assert [item.chunk_id for item in results] == ["a", "b", "c"]
    assert [item.rank for item in results] == [1, 2, 3]
    assert [item.original_rrf_rank for item in results] == [2, 1, 3]
    assert [item.rerank_score for item in results] == [0.9, 0.4, 0.4]
    assert [item.rerank_truncated for item in results] == [True, False, False]
    assert results[1].rrf_score == candidates[0].rrf_score
    assert results[1].dense_score == candidates[0].dense_score
    assert results[1].bm25_score == candidates[0].bm25_score
    assert scorer.calls == [("create index", ["passage", "passage", "passage"])]
    assert [item.rank for item in candidates] == [1, 2, 3]


def test_rerank_hits_uses_original_rank_then_chunk_id_for_equal_logits() -> None:
    candidates = [_hit(2, "z"), _hit(1, "b"), _hit(1, "a")]
    scorer = FixedScorer([PairScore(1.0, False)] * 3)

    results = rerank_hits("query", candidates, scorer, top_k=2)

    assert [item.chunk_id for item in results] == ["a", "b"]


@pytest.mark.parametrize(
    ("query", "candidates", "scores", "message"),
    [
        (" ", [_hit(1, "a")], [PairScore(1.0, False)], "query"),
        ("q", [_hit(1, "a", " ")], [PairScore(1.0, False)], "passage"),
        ("q", [_hit(1, "a"), _hit(2, "a")], [PairScore(1.0, False)] * 2, "duplicate"),
        ("q", [_hit(1, "a")], [], "count"),
        ("q", [_hit(1, "a")], [PairScore(float("nan"), False)], "finite"),
    ],
)
def test_rerank_hits_rejects_invalid_inputs(
    query: str,
    candidates: list[HybridSearchHit],
    scores: list[PairScore],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError, RerankerError), match=message):
        rerank_hits(query, candidates, FixedScorer(scores), top_k=1)


@dataclass
class FakeEncoding:
    ids: list[int]
    attention_mask: list[int]
    overflowing: list[object]


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.truncation: dict[str, object] | None = None
        self.padding: dict[str, object] | None = None

    def token_to_id(self, token: str) -> int | None:
        return 1 if token == "<pad>" else None

    def enable_truncation(self, **kwargs: object) -> None:
        self.truncation = kwargs

    def enable_padding(self, **kwargs: object) -> None:
        self.padding = kwargs

    def encode_batch(self, pairs: list[tuple[str, str]]) -> list[FakeEncoding]:
        self.calls.append(pairs)
        return [
            FakeEncoding(
                ids=[0, index + 3, 2],
                attention_mask=[1, 1, 1],
                overflowing=[object()] if "truncate" in passage else [],
            )
            for index, (_, passage) in enumerate(pairs)
        ]


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    def __init__(self, *, input_names: tuple[str, ...] = ("input_ids", "attention_mask")) -> None:
        self.input_names = input_names
        self.providers: list[str] = []
        self.model_path = ""
        self.calls: list[dict[str, np.ndarray]] = []
        self.non_finite = False
        self.wrong_shape = False

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput(name) for name in self.input_names]

    def run(self, outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        del outputs
        self.calls.append(inputs)
        count = inputs["input_ids"].shape[0]
        if self.wrong_shape:
            return [np.zeros((count, 2), dtype=np.float32)]
        logits = np.arange(count, dtype=np.float32).reshape(count, 1)
        if self.non_finite:
            logits[0, 0] = np.inf
        return [logits]


def _reranker(
    tmp_path: Path,
    *,
    session: FakeSession | None = None,
) -> tuple[CrossEncoderReranker, FakeTokenizer, FakeSession]:
    model_root = tmp_path / "model"
    (model_root / "onnx").mkdir(parents=True)
    (model_root / "onnx" / "model.onnx").write_bytes(b"fake")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    tokenizer = FakeTokenizer()
    fake_session = session or FakeSession()

    def tokenizer_factory(path: str) -> FakeTokenizer:
        assert path == str(tokenizer_path)
        return tokenizer

    def session_factory(path: str, *, providers: list[str]) -> FakeSession:
        fake_session.model_path = path
        fake_session.providers = providers
        return fake_session

    reranker = CrossEncoderReranker(
        VerifiedModelAssets(model_root.resolve(), 1, 4),
        tokenizer_path,
        replace(CONFIG, batch_size=2),
        tokenizer_factory=tokenizer_factory,
        session_factory=session_factory,
    )
    return reranker, tokenizer, fake_session


def test_cross_encoder_batches_pairs_and_reports_truncation(tmp_path: Path) -> None:
    reranker, tokenizer, session = _reranker(tmp_path)

    scores = reranker.score_pairs("query", ["one", "truncate two", "three"])

    assert tokenizer.truncation == {"max_length": 512, "strategy": "only_second"}
    assert tokenizer.padding == {"pad_id": 1, "pad_token": "<pad>"}
    assert tokenizer.calls == [
        [("query", "one"), ("query", "truncate two")],
        [("query", "three")],
    ]
    assert [item.logit for item in scores] == [0.0, 1.0, 0.0]
    assert [item.truncated for item in scores] == [False, True, False]
    assert len(session.calls) == 2
    assert session.providers == ["CPUExecutionProvider"]
    assert Path(session.model_path).parts[-2:] == ("onnx", "model.onnx")


def test_cross_encoder_rejects_wrong_inputs_and_outputs(tmp_path: Path) -> None:
    with pytest.raises(RerankerError, match="inputs"):
        _reranker(tmp_path, session=FakeSession(input_names=("input_ids",)))

    wrong, _, wrong_session = _reranker(tmp_path / "wrong")
    wrong_session.wrong_shape = True
    with pytest.raises(RerankerError, match="shape"):
        wrong.score_pairs("query", ["passage"])

    non_finite, _, non_finite_session = _reranker(tmp_path / "nan")
    non_finite_session.non_finite = True
    with pytest.raises(RerankerError, match="non-finite"):
        non_finite.score_pairs("query", ["passage"])
