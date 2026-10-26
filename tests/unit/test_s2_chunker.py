import hashlib
import json
import os
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

from docguard.processing.chunker import ChunkingError, chunk_document
from docguard.processing.config import ChunkingConfig, ModelSpec
from docguard.processing.models import DocumentRecord, SectionBlock, make_document_id
from docguard.processing.token_budget import TokenBudget, TokenBudgetError

URL = "https://www.postgresql.org/docs/17/sql-select.html"


class WordBudget:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum

    def count(self, text: str) -> int:
        return len(text.split())

    def validate_chunk(self, text: str) -> int:
        count = self.count(text)
        if count > self.maximum:
            raise TokenBudgetError(f"chunk has {count} tokens")
        return count


class ImpossibleBudget(WordBudget):
    def count(self, text: str) -> int:
        return 100 if text else 0


def _config(
    *, max_tokens: int = 8, overlap: int = 3, minimum: int = 2
) -> ChunkingConfig:
    model = ModelSpec("test/model", "a" * 40, 512)
    return ChunkingConfig(
        schema_version=1,
        chunking_version="s2-chunk-v1",
        max_chunk_tokens=max_tokens,
        overlap_tokens=overlap,
        min_chunk_tokens=minimum,
        query_budget=1,
        special_tokens_budget=1,
        safety_margin=1,
        html_normalization_version="postgresql-html-v1",
        id_rule_version="sha256-nul-v1",
        embedding=model,
        reranker=model,
    )


def _document() -> DocumentRecord:
    return DocumentRecord(
        document_id=make_document_id("17.11", URL),
        postgresql_version="17.11",
        topic="sql_commands",
        canonical_url=URL,
        title="SELECT",
        content_sha256="a" * 64,
    )


def _block(text: str, *, anchor: str = "a", kind: str = "paragraph") -> SectionBlock:
    return SectionBlock(
        kind=kind,  # type: ignore[arg-type]
        anchor=anchor,
        section_path=("T",),
        text=text,
    )


def _tokenizer_bytes(*, split_punctuation: bool, add_special_tokens: bool = False) -> bytes:
    vocabulary = {
        "[UNK]": 0,
        "alpha": 1,
        "-": 2,
        "beta": 3,
        "alpha-beta": 4,
        "[CLS]": 5,
        "[SEP]": 6,
    }
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = (
        pre_tokenizers.Whitespace()
        if split_punctuation
        else pre_tokenizers.WhitespaceSplit()
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    if add_special_tokens:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[("[CLS]", 5), ("[SEP]", 6)],
        )
    return tokenizer.to_str().encode("utf-8")


def _write_tokenizer_manifest(root: Path, *, add_special_tokens: bool = False) -> Path:
    entries = []
    for index, payload in enumerate(
        (
            _tokenizer_bytes(
                split_punctuation=False, add_special_tokens=add_special_tokens
            ),
            _tokenizer_bytes(
                split_punctuation=True, add_special_tokens=add_special_tokens
            ),
        )
    ):
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path("data", "tokenizers", digest, "tokenizer.json")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        entries.append(
            {
                "name": f"test/model-{index}",
                "revision": str(index) * 40,
                "asset_path": relative.as_posix(),
                "size_bytes": len(payload),
                "sha256": digest,
            }
        )
    manifest = root / "data" / "tokenizers" / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "models": entries}), encoding="utf-8"
    )
    return manifest


def test_token_budget_uses_maximum_count_across_frozen_tokenizers(tmp_path: Path) -> None:
    budget = TokenBudget.from_manifest(_write_tokenizer_manifest(tmp_path), tmp_path)

    assert budget.count("alpha-beta") == 3


def test_token_budget_count_includes_configured_special_tokens(tmp_path: Path) -> None:
    budget = TokenBudget.from_manifest(
        _write_tokenizer_manifest(tmp_path, add_special_tokens=True), tmp_path
    )

    assert budget.count("alpha") == 3


def test_token_budget_rejects_hash_mismatch_and_hardlinked_asset(tmp_path: Path) -> None:
    manifest_path = _write_tokenizer_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_asset = tmp_path / manifest["models"][0]["asset_path"]
    first_asset.write_bytes(first_asset.read_bytes() + b" ")
    with pytest.raises(TokenBudgetError, match="SHA-256|size"):
        TokenBudget.from_manifest(manifest_path, tmp_path)

    manifest_path = _write_tokenizer_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_asset = tmp_path / manifest["models"][0]["asset_path"]
    os.link(first_asset, tmp_path / "hardlink.json")
    with pytest.raises(TokenBudgetError, match="hardlink"):
        TokenBudget.from_manifest(manifest_path, tmp_path)


def test_token_budget_enforces_384_token_limit(tmp_path: Path) -> None:
    budget = TokenBudget.from_manifest(_write_tokenizer_manifest(tmp_path), tmp_path)

    assert budget.validate_chunk("alpha " * 384) == 384
    with pytest.raises(TokenBudgetError, match="385"):
        budget.validate_chunk("alpha " * 385)


def test_chunker_packs_complete_blocks_with_target_overlap() -> None:
    chunks = chunk_document(
        _document(),
        (_block("a b c"), _block("d e f"), _block("g h i")),
        _config(),
        WordBudget(8),
    )

    assert [chunk.text for chunk in chunks] == [
        "T\n\na b c\n\nd e f",
        "T\n\nd e f\n\ng h i",
    ]
    assert [chunk.token_count for chunk in chunks] == [7, 7]
    assert [chunk.ordinal for chunk in chunks] == [0, 1]
    assert len({chunk.chunk_id for chunk in chunks}) == 2


def test_chunker_never_mixes_anchors_and_ordinals_restart_per_anchor() -> None:
    chunks = chunk_document(
        _document(),
        (_block("one", anchor="a"), _block("two", anchor="b")),
        _config(),
        WordBudget(8),
    )

    assert [(chunk.anchor, chunk.ordinal) for chunk in chunks] == [("a", 0), ("b", 0)]
    assert chunks == chunk_document(
        _document(),
        (_block("one", anchor="a"), _block("two", anchor="b")),
        _config(),
        WordBudget(8),
    )


def test_chunker_splits_long_paragraph_on_sentences_then_windows() -> None:
    chunks = chunk_document(
        _document(),
        (_block("one two three. four five six. seven eight nine ten eleven."),),
        _config(max_tokens=6, overlap=1),
        WordBudget(6),
    )

    assert len(chunks) >= 3
    assert all(chunk.token_count <= 6 for chunk in chunks)
    assert "one two three." in chunks[0].text
    assert "eleven." in chunks[-1].text


def test_chunker_repeats_table_header_and_code_explanation() -> None:
    table_chunks = chunk_document(
        _document(),
        (_block("head one | head two\na | b\nc | d\ne | f", kind="table"),),
        _config(max_tokens=9, overlap=1),
        WordBudget(9),
    )
    code_chunks = chunk_document(
        _document(),
        (
            _block("Use this example."),
            _block("line one\nline two\nline three\nline four", kind="code"),
        ),
        _config(max_tokens=8, overlap=1),
        WordBudget(8),
    )

    assert len(table_chunks) > 1
    assert all("head one | head two" in chunk.text for chunk in table_chunks)
    assert len(code_chunks) > 1
    assert all("Use this example." in chunk.text for chunk in code_chunks[1:])
    assert all(chunk.token_count <= 9 for chunk in table_chunks)
    assert all(chunk.token_count <= 8 for chunk in code_chunks)


def test_chunker_splits_oversized_table_row_without_losing_header() -> None:
    chunks = chunk_document(
        _document(),
        (_block("Header\none two three four five six seven eight", kind="table"),),
        _config(max_tokens=6, overlap=1),
        WordBudget(6),
    )

    assert len(chunks) >= 2
    assert all("Header" in chunk.text for chunk in chunks)
    combined = " ".join(chunk.text.split("Header", 1)[1] for chunk in chunks)
    expected_words = ("one", "two", "three", "four", "five", "six", "seven", "eight")
    assert all(word in combined for word in expected_words)
    assert all(chunk.token_count <= 6 for chunk in chunks)


def test_chunker_fails_closed_when_even_minimal_content_cannot_fit() -> None:
    with pytest.raises(ChunkingError, match="a.*token|budget"):
        chunk_document(
            _document(),
            (_block("impossible"),),
            _config(),
            ImpossibleBudget(8),
        )

