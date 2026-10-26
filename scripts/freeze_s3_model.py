"""Freeze the exact ONNX files for the S3 E5 embedding model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import hf_hub_download

from docguard.fs_safety import (
    FileConflictError,
    FileSafetyError,
    atomic_rename_directory_noreplace,
    create_stage,
    ensure_directory,
    exclusive_lock,
    validate_directory,
    write_stage_file,
)
from docguard.retrieval.model_assets import ModelAssetError, verify_model_assets

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_FILES = {
    "config.json": (
        655,
        "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    ),
    "onnx/config.json": (
        653,
        "bbb7c1333fc4b3e27fbc9cd5d2070aabcc1d4dfb99917c3633e772f97545a6b6",
    ),
    "onnx/model.onnx": (
        470268510,
        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    ),
}


def _stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _manifest_payload() -> dict[str, object]:
    files = [
        {"path": path, "size_bytes": size, "sha256": digest}
        for path, (size, digest) in sorted(MODEL_FILES.items())
    ]
    return {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "revision": MODEL_REVISION,
        "files": files,
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }


def _verify_payload(path: str, payload: bytes) -> None:
    expected_size, expected_digest = MODEL_FILES[path]
    if len(payload) != expected_size:
        raise ModelAssetError(f"downloaded model size mismatch: {path}")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ModelAssetError(f"downloaded model SHA-256 mismatch: {path}")


def _write_nested_stage(stage: Path, relative: str, payload: bytes) -> None:
    parts = relative.split("/")
    directory = stage
    for part in parts[:-1]:
        directory = directory / part
        ensure_directory(directory)
    write_stage_file(directory, parts[-1], payload)


def _verify_hardcoded_assets(model_root: Path) -> None:
    validate_directory(model_root)
    actual = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file()
    }
    if actual != set(MODEL_FILES):
        raise ModelAssetError("model file set does not match the hardcoded contract")
    for relative in sorted(MODEL_FILES):
        payload = (model_root / Path(relative)).read_bytes()
        _verify_payload(relative, payload)


def freeze_model(
    model_root: Path,
    manifest_path: Path,
    *,
    download: bool,
) -> None:
    manifest_bytes = _stable_json_bytes(_manifest_payload())
    ensure_directory(model_root.parent)
    ensure_directory(manifest_path.parent)
    with exclusive_lock(model_root.parent, model_root.name):
        if os.path.lexists(model_root):
            _verify_hardcoded_assets(model_root)
        else:
            if not download:
                raise ModelAssetError(
                    "model assets are absent; rerun with --download to access the fixed revision"
                )
            stage, _ = create_stage(model_root.parent, f".{model_root.name}.stage-")
            try:
                for relative in sorted(MODEL_FILES):
                    cached = Path(
                        hf_hub_download(
                            repo_id=MODEL_NAME,
                            filename=relative,
                            revision=MODEL_REVISION,
                        )
                    )
                    payload = cached.read_bytes()
                    _verify_payload(relative, payload)
                    _write_nested_stage(stage, relative, payload)
                _verify_hardcoded_assets(stage)
                atomic_rename_directory_noreplace(stage, model_root)
            except Exception as error:
                raise ModelAssetError(
                    f"model staging retained for manual inspection: {stage}: {error}"
                ) from error

        if os.path.lexists(manifest_path):
            if manifest_path.read_bytes() != manifest_bytes:
                raise ModelAssetError("existing model manifest differs from fixed contract")
        else:
            write_stage_file(manifest_path.parent, manifest_path.name, manifest_bytes)
        verify_model_assets(model_root, manifest_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--model-root", type=Path, default=Path("data/models/s3-e5-small")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/s3-embedding-model.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        freeze_model(args.model_root, args.manifest, download=args.download)
        return 0
    except (FileConflictError, FileSafetyError, ModelAssetError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
