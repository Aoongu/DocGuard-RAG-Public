import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "retrieval" / "s3-dense-v1.json"


def _modules():
    config_module = importlib.import_module("docguard.retrieval.config")
    assets_module = importlib.import_module("docguard.retrieval.model_assets")
    return config_module, assets_module


def _asset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"hidden_size":384}\n', encoding="utf-8")
    pool = model_root / "1_Pooling"
    pool.mkdir()
    (pool / "config.json").write_text('{"pooling_mode_mean_tokens":true}\n', encoding="utf-8")
    files = []
    for path in sorted(item for item in model_root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(model_root).as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "intfloat/multilingual-e5-small",
                "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                "files": files,
                "file_count": len(files),
                "total_size_bytes": sum(item["size_bytes"] for item in files),
            }
        ),
        encoding="utf-8",
    )
    return model_root, manifest


def test_official_dense_config_is_strict_and_frozen() -> None:
    config_module, _ = _modules()

    config = config_module.load_dense_config(CONFIG)

    assert config.index_version == "s3-dense-v1"
    assert config.dense_enabled is True
    assert config.model.name == "intfloat/multilingual-e5-small"
    assert config.model.revision == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert config.model.embedding_dimension == 384
    assert config.model.max_length == 512
    assert config.model.query_prefix == "query: "
    assert config.model.passage_prefix == "passage: "
    assert config.model.normalize_embeddings is True
    assert config.batch_size == 32
    assert config.collection_name == "docguard_postgresql_s2_chunk_v1_e5_small"
    assert config.distance == "Cosine"
    assert config.default_top_k == 5
    assert config.chunking_version == "s2-chunk-v1"
    assert config.s2_manifest_sha256 == (
        "0559ee4d75d6079fc2859f8e09910570c4f37f631511f0a9f4348ae7023911d2"
    )
    assert config.chunks_sha256 == (
        "6aad4667593d6519657cd1823b713983e09ed9f819e48246e0db9c18f0694964"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("index_version", "moving", "index_version"),
        ("dense_enabled", 1, "dense_enabled"),
        ("embedding_dimension", 768, "embedding_dimension"),
        ("revision", "main", "revision"),
        ("distance", "Euclid", "distance"),
        ("default_top_k", 0, "default_top_k"),
        ("chunks_sha256", "bad", "chunks_sha256"),
    ],
)
def test_dense_config_rejects_contract_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config_module, _ = _modules()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    if field in payload["model"]:
        payload["model"][field] = value
    else:
        payload[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        config_module.load_dense_config(path)


def test_dense_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_module, _ = _modules()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["surprise"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        config_module.load_dense_config(path)


def test_model_assets_verify_exact_files_sizes_and_hashes(tmp_path: Path) -> None:
    _, assets_module = _modules()
    model_root, manifest = _asset_fixture(tmp_path)

    verified = assets_module.verify_model_assets(model_root, manifest)

    assert verified.model_root == model_root.resolve()
    assert verified.file_count == 2
    assert verified.total_size_bytes > 0


@pytest.mark.parametrize("mutation", ["missing", "extra", "content"])
def test_model_assets_reject_file_set_and_content_drift(
    tmp_path: Path, mutation: str
) -> None:
    _, assets_module = _modules()
    model_root, manifest = _asset_fixture(tmp_path)
    if mutation == "missing":
        (model_root / "config.json").unlink()
    elif mutation == "extra":
        (model_root / "extra.bin").write_bytes(b"extra")
    else:
        (model_root / "config.json").write_bytes(b"tampered")

    with pytest.raises(assets_module.ModelAssetError):
        assets_module.verify_model_assets(model_root, manifest)


def test_model_assets_reject_hardlinks(tmp_path: Path) -> None:
    _, assets_module = _modules()
    model_root, manifest = _asset_fixture(tmp_path)
    os.link(model_root / "config.json", tmp_path / "hardlink.json")

    with pytest.raises(assets_module.ModelAssetError, match="unsafe model file"):
        assets_module.verify_model_assets(model_root, manifest)


def test_model_assets_reject_symlinks_when_supported(tmp_path: Path) -> None:
    _, assets_module = _modules()
    model_root, manifest = _asset_fixture(tmp_path)
    original = model_root / "config.json"
    target = tmp_path / "target.json"
    target.write_bytes(original.read_bytes())
    original.unlink()
    try:
        original.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(assets_module.ModelAssetError, match="unsafe model file"):
        assets_module.verify_model_assets(model_root, manifest)
