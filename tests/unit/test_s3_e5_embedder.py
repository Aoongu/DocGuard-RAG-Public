from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from docguard.retrieval.config import load_dense_config
from docguard.retrieval.e5_embedder import E5Embedder, E5EmbeddingError
from docguard.retrieval.model_assets import VerifiedModelAssets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_dense_config(
    PROJECT_ROOT / "configs" / "retrieval" / "s3-dense-v1.json"
)


@dataclass
class FakeEncoding:
    ids: list[int]
    attention_mask: list[int]


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.truncation: dict | None = None
        self.padding: dict | None = None

    def token_to_id(self, token: str) -> int | None:
        return 1 if token == "<pad>" else None

    def enable_truncation(self, **kwargs: object) -> None:
        self.truncation = kwargs

    def enable_padding(self, **kwargs: object) -> None:
        self.padding = kwargs

    def encode_batch(self, texts: list[str]) -> list[FakeEncoding]:
        self.calls.append(texts)
        return [FakeEncoding([0, index + 2, 2], [1, 1, 1]) for index, _ in enumerate(texts)]


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    def __init__(self, dimension: int = 384, non_finite: bool = False) -> None:
        self.providers: list[str] = []
        self.model_path = ""
        self.dimension = dimension
        self.non_finite = non_finite

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("input_ids"), FakeInput("attention_mask")]

    def run(self, outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        del outputs
        ids = inputs["input_ids"]
        hidden = np.repeat(ids[:, :, None], self.dimension, axis=2).astype(np.float32)
        offsets = np.arange(self.dimension, dtype=np.float32)[None, None, :] / 1000
        hidden = hidden + offsets
        if self.non_finite:
            hidden[0, 0, 0] = np.nan
        return [hidden]


def _embedder(
    tmp_path: Path,
    *,
    dimension: int = 384,
    non_finite: bool = False,
) -> tuple[E5Embedder, FakeTokenizer, FakeSession]:
    model_root = tmp_path / "model"
    (model_root / "onnx").mkdir(parents=True)
    (model_root / "onnx" / "model.onnx").write_bytes(b"fake")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    tokenizer = FakeTokenizer()
    session = FakeSession(dimension=dimension, non_finite=non_finite)

    def tokenizer_factory(path: str) -> FakeTokenizer:
        assert path == str(tokenizer_path)
        return tokenizer

    def session_factory(path: str, *, providers: list[str]) -> FakeSession:
        session.model_path = path
        session.providers = providers
        return session

    embedder = E5Embedder(
        VerifiedModelAssets(model_root.resolve(), 1, 4),
        tokenizer_path,
        CONFIG,
        tokenizer_factory=tokenizer_factory,
        session_factory=session_factory,
    )
    return embedder, tokenizer, session


@pytest.mark.asyncio
async def test_e5_embedder_uses_distinct_prefixes_cpu_and_normalized_vectors(
    tmp_path: Path,
) -> None:
    embedder, tokenizer, session = _embedder(tmp_path)

    passages = await embedder.embed_passages(["CREATE INDEX"])
    queries = await embedder.embed_queries(["如何创建索引？"])

    assert tokenizer.calls == [
        ["passage: CREATE INDEX"],
        ["query: 如何创建索引？"],
    ]
    assert tokenizer.truncation == {"max_length": 512}
    assert tokenizer.padding == {"pad_id": 1, "pad_token": "<pad>"}
    assert session.providers == ["CPUExecutionProvider"]
    assert Path(session.model_path).parts[-2:] == ("onnx", "model.onnx")
    assert len(passages[0]) == len(queries[0]) == 384
    assert np.linalg.norm(passages[0]) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(queries[0]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_e5_embedder_batches_and_empty_input_is_a_noop(tmp_path: Path) -> None:
    embedder, tokenizer, _ = _embedder(tmp_path)
    texts = [f"text {index}" for index in range(33)]

    vectors = await embedder.embed_passages(texts)
    empty = await embedder.embed_queries([])

    assert len(vectors) == 33
    assert [len(call) for call in tokenizer.calls] == [32, 1]
    assert empty == []


@pytest.mark.asyncio
async def test_e5_embedder_rejects_ambiguous_embed_text(tmp_path: Path) -> None:
    embedder, _, _ = _embedder(tmp_path)

    with pytest.raises(TypeError, match="embed_passages or embed_queries"):
        await embedder.embed_text(["ambiguous"])


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", " ", "bad\0text"])
async def test_e5_embedder_rejects_invalid_text(tmp_path: Path, text: str) -> None:
    embedder, _, _ = _embedder(tmp_path)

    with pytest.raises(ValueError, match="non-empty"):
        await embedder.embed_queries([text])


@pytest.mark.asyncio
async def test_e5_embedder_rejects_wrong_dimension(tmp_path: Path) -> None:
    embedder, _, _ = _embedder(tmp_path, dimension=16)

    with pytest.raises(E5EmbeddingError, match="dimension"):
        await embedder.embed_passages(["text"])


@pytest.mark.asyncio
async def test_e5_embedder_rejects_non_finite_output(tmp_path: Path) -> None:
    embedder, _, _ = _embedder(tmp_path, non_finite=True)

    with pytest.raises(E5EmbeddingError, match="non-finite"):
        await embedder.embed_queries(["text"])


@pytest.mark.asyncio
async def test_e5_embedder_reports_ragbits_vector_size(tmp_path: Path) -> None:
    embedder, _, _ = _embedder(tmp_path)

    vector_size = await embedder.get_vector_size()

    assert vector_size.size == 384
    assert vector_size.is_sparse is False
