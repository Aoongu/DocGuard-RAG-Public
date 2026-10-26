from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "freeze_s6_reranker.py"
SPEC = importlib.util.spec_from_file_location("freeze_s6_reranker", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _fixture_files(tmp_path: Path) -> dict[str, tuple[int, str]]:
    source = tmp_path / "source"
    (source / "onnx").mkdir(parents=True)
    payloads = {
        "config.json": b'{"model_type":"xlm-roberta"}\n',
        "onnx/model.onnx": b"fixed-onnx-fixture",
    }
    for relative, payload in payloads.items():
        path = source / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    MODULE.MODEL_FILES = {
        relative: (len(payload), hashlib.sha256(payload).hexdigest())
        for relative, payload in payloads.items()
    }
    return MODULE.MODEL_FILES


def test_s6_freezer_constants_pin_official_revision() -> None:
    assert MODULE.MODEL_NAME == "BAAI/bge-reranker-base"
    assert MODULE.MODEL_REVISION == "2cfc18c9415c912f9d8155881c133215df768a70"
    assert MODULE.MODEL_FILES == {
        "config.json": (
            799,
            "289adf7ada1eb6b4afa7589a48a032d45a076cf2e46dcdb3b4cabc33be14f708",
        ),
        "onnx/model.onnx": (
            1112459588,
            "15b9a8c3da82eddf263df571281166e00e9308fe19d077084b642ebfcaf06d2b",
        ),
    }


def test_s6_freezer_is_offline_by_default(tmp_path: Path) -> None:
    with pytest.raises(MODULE.ModelAssetError, match="--download"):
        MODULE.freeze_model(
            tmp_path / "model",
            tmp_path / "manifest.json",
            download=False,
        )


def test_s6_freezer_downloads_only_fixed_files_and_reuses_verified_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture_files(tmp_path)
    source = tmp_path / "source"
    calls: list[tuple[str, str, str]] = []

    def fake_download(*, repo_id: str, filename: str, revision: str) -> str:
        calls.append((repo_id, filename, revision))
        return str(source / Path(filename))

    monkeypatch.setattr(MODULE, "hf_hub_download", fake_download)
    model_root = tmp_path / "published" / "model"
    manifest = tmp_path / "published" / "manifest.json"

    MODULE.freeze_model(model_root, manifest, download=True)
    first_manifest = manifest.read_bytes()
    MODULE.freeze_model(model_root, manifest, download=False)

    assert calls == [
        (MODULE.MODEL_NAME, relative, MODULE.MODEL_REVISION)
        for relative in sorted(MODULE.MODEL_FILES)
    ]
    assert manifest.read_bytes() == first_manifest
    assert {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file()
    } == set(MODULE.MODEL_FILES)
    payload = json.loads(first_manifest)
    assert payload["model_name"] == MODULE.MODEL_NAME
    assert payload["revision"] == MODULE.MODEL_REVISION
    assert payload["file_count"] == 2


def test_s6_freezer_rejects_tampered_existing_asset(tmp_path: Path) -> None:
    _fixture_files(tmp_path)
    model_root = tmp_path / "model"
    (model_root / "onnx").mkdir(parents=True)
    (model_root / "config.json").write_bytes(b"tampered")
    (model_root / "onnx" / "model.onnx").write_bytes(b"fixed-onnx-fixture")

    with pytest.raises(MODULE.ModelAssetError, match="size mismatch"):
        MODULE.freeze_model(
            model_root,
            tmp_path / "manifest.json",
            download=False,
        )


def test_generic_verifier_accepts_explicit_s6_identity(tmp_path: Path) -> None:
    from docguard.retrieval.model_assets import ModelAssetError, verify_model_assets

    files = _fixture_files(tmp_path)
    model_root = tmp_path / "source"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": MODULE.MODEL_NAME,
                "revision": MODULE.MODEL_REVISION,
                "files": [
                    {"path": path, "size_bytes": size, "sha256": digest}
                    for path, (size, digest) in sorted(files.items())
                ],
                "file_count": 2,
                "total_size_bytes": sum(size for size, _ in files.values()),
            }
        ),
        encoding="utf-8",
    )

    verified = verify_model_assets(
        model_root,
        manifest,
        expected_model_name=MODULE.MODEL_NAME,
        expected_revision=MODULE.MODEL_REVISION,
    )
    assert verified.file_count == 2

    with pytest.raises(ModelAssetError, match="unexpected model name"):
        verify_model_assets(
            model_root,
            manifest,
            expected_model_name="wrong/model",
            expected_revision=MODULE.MODEL_REVISION,
        )
