import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from docguard.processing.models import ChunkRecord, make_chunk_id, make_document_id
from docguard.retrieval.pipeline import (
    DenseIndexConflictError,
    DenseIndexError,
    build_index,
    verify_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CONFIG = PROJECT_ROOT / "configs" / "retrieval" / "s3-dense-v1.json"


class FakeEmbedder:
    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        prefix = [
            float("alpha" in lowered),
            float("beta" in lowered),
            float("gamma" in lowered),
        ]
        if prefix == [0.0, 0.0, 0.0]:
            prefix[2] = 1.0
        return prefix + [0.0] * 381


class FailingEmbedder(FakeEmbedder):
    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise RuntimeError("controlled embedding failure")


def _chunk(index: int, version: str, text: str) -> ChunkRecord:
    major = version.split(".", 1)[0]
    url = f"https://www.postgresql.org/docs/{major}/sql-select.html"
    anchor = f"SECTION-{index}"
    return ChunkRecord(
        schema_version=1,
        chunk_id=make_chunk_id("s2-chunk-v1", version, url, anchor, index),
        document_id=make_document_id(version, url),
        postgresql_version=version,
        topic="sql_commands",
        canonical_url=url,
        anchor=anchor,
        section_path=("SELECT", f"Section {index}"),
        ordinal=index,
        text=text,
        token_count=3,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        chunking_version="s2-chunk-v1",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    chunks = tmp_path / "chunks.jsonl"
    records = [
        _chunk(1, "17.11", "alpha primary"),
        _chunk(2, "17.11", "alpha secondary"),
        _chunk(3, "18.6", "beta content"),
        _chunk(4, "18.6", "gamma content"),
    ]
    chunks.write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
        newline="\n",
    )
    config_payload = json.loads(OFFICIAL_CONFIG.read_text(encoding="utf-8"))
    config_payload["chunks_sha256"] = hashlib.sha256(chunks.read_bytes()).hexdigest()
    config = tmp_path / "config.json"
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    model_manifest = tmp_path / "model-manifest.json"
    model_manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "indexes" / "s3-dense-v1"
    return chunks, config, model_manifest, output


@pytest.mark.asyncio
async def test_index_build_verify_and_idempotent_reuse(tmp_path: Path) -> None:
    chunks, config, model_manifest, output = _fixture(tmp_path)

    first = await build_index(
        chunks,
        output,
        config_path=config,
        model_manifest_path=model_manifest,
        embedder=FakeEmbedder(),
    )
    verified = await verify_index(
        chunks,
        output,
        config_path=config,
        model_manifest_path=model_manifest,
        embedder=FakeEmbedder(),
    )
    second = await build_index(
        chunks,
        output,
        config_path=config,
        model_manifest_path=model_manifest,
        embedder=FakeEmbedder(),
    )

    assert first.status == "built"
    assert verified.status == "verified"
    assert second.status == "skipped"
    assert first.point_count == verified.point_count == second.point_count == 4
    assert first.version_counts == {"17.11": 2, "18.6": 2}
    assert {path.name for path in output.iterdir()} == {"build-info.json", "qdrant"}
    build_info = json.loads((output / "build-info.json").read_text(encoding="utf-8"))
    assert build_info["point_count"] == 4
    assert build_info["vector_size"] == 384
    assert build_info["distance"] == "Cosine"


@pytest.mark.asyncio
async def test_index_conflict_is_not_overwritten(tmp_path: Path) -> None:
    chunks, config, model_manifest, output = _fixture(tmp_path)
    await build_index(
        chunks,
        output,
        config_path=config,
        model_manifest_path=model_manifest,
        embedder=FakeEmbedder(),
    )
    info = output / "build-info.json"
    info.write_text("{}\n", encoding="utf-8")
    before = info.read_bytes()

    with pytest.raises(DenseIndexConflictError, match="build-info"):
        await verify_index(
            chunks,
            output,
            config_path=config,
            model_manifest_path=model_manifest,
            embedder=FakeEmbedder(),
        )

    assert info.read_bytes() == before


@pytest.mark.asyncio
async def test_index_rejects_input_hash_drift_before_creating_output(
    tmp_path: Path,
) -> None:
    chunks, config, model_manifest, output = _fixture(tmp_path)
    chunks.write_bytes(chunks.read_bytes() + b"tampered\n")

    with pytest.raises(DenseIndexError, match="chunks SHA-256"):
        await build_index(
            chunks,
            output,
            config_path=config,
            model_manifest_path=model_manifest,
            embedder=FakeEmbedder(),
        )

    assert not output.exists()


@pytest.mark.asyncio
async def test_failed_build_retains_stage_for_manual_inspection(tmp_path: Path) -> None:
    chunks, config, model_manifest, output = _fixture(tmp_path)

    with pytest.raises(DenseIndexError, match="staging retained") as error:
        await build_index(
            chunks,
            output,
            config_path=config,
            model_manifest_path=model_manifest,
            embedder=FailingEmbedder(),
        )

    retained = Path(str(error.value).split("staging retained for manual inspection: ", 1)[1].split(": controlled", 1)[0])
    assert retained.is_dir()
    assert not output.exists()
