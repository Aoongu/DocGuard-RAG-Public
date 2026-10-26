"""Fail-closed verification for the locally cached S3 embedding model."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from docguard.fs_safety import (
    FileSafetyError,
    validate_directory,
    validate_regular_file,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_MANIFEST_FIELDS = frozenset(
    {"schema_version", "model_name", "revision", "files", "file_count", "total_size_bytes"}
)
_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256"})


class ModelAssetError(RuntimeError):
    """Raised when model assets do not match the frozen manifest."""


@dataclass(frozen=True, slots=True)
class VerifiedModelAssets:
    model_root: Path
    file_count: int
    total_size_bytes: int


def _strict_fields(data: dict, expected: frozenset[str], label: str) -> None:
    if set(data) != expected:
        raise ModelAssetError(f"{label} fields do not match the frozen schema")


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ModelAssetError("model file path must be a non-empty POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
        raise ModelAssetError(f"unsafe model file path: {value}")
    return posix.as_posix()


def _walk_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            validate_directory(directory)
            entries = list(os.scandir(directory))
        except (OSError, FileSafetyError) as error:
            raise ModelAssetError(f"unsafe model directory: {directory}") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                if entry.is_dir(follow_symlinks=False):
                    validate_directory(path)
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    validate_regular_file(path)
                    relative = path.relative_to(root).as_posix()
                    files[relative] = path
                else:
                    raise ModelAssetError(f"unsafe model file: {path}")
            except FileSafetyError as error:
                raise ModelAssetError(f"unsafe model file: {path}") from error
    return files


def verify_model_assets(
    model_root: Path,
    manifest_path: Path,
    *,
    expected_model_name: str = "intfloat/multilingual-e5-small",
    expected_revision: str | None = None,
) -> VerifiedModelAssets:
    """Verify an exact local model directory without accessing the network."""
    try:
        validate_directory(model_root)
        validate_regular_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, FileSafetyError, json.JSONDecodeError) as error:
        raise ModelAssetError("unable to load model assets or manifest") from error
    if not isinstance(manifest, dict):
        raise ModelAssetError("model manifest must be an object")
    _strict_fields(manifest, _MANIFEST_FIELDS, "model manifest")
    if manifest["schema_version"] != 1:
        raise ModelAssetError("model manifest schema_version must be 1")
    if manifest["model_name"] != expected_model_name:
        raise ModelAssetError("unexpected model name")
    if not isinstance(manifest["revision"], str) or _REVISION_RE.fullmatch(
        manifest["revision"]
    ) is None:
        raise ModelAssetError("model revision must be a commit SHA")
    if expected_revision is not None and manifest["revision"] != expected_revision:
        raise ModelAssetError("unexpected model revision")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise ModelAssetError("model manifest files must be a non-empty list")

    expected: dict[str, tuple[int, str]] = {}
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise ModelAssetError("model file record must be an object")
        _strict_fields(item, _FILE_FIELDS, "model file record")
        relative = _safe_relative_path(item["path"])
        size = item["size_bytes"]
        digest = item["sha256"]
        if type(size) is not int or size <= 0:
            raise ModelAssetError("model file size must be positive")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ModelAssetError("model file sha256 must be lowercase")
        if relative in expected:
            raise ModelAssetError(f"duplicate model file: {relative}")
        expected[relative] = size, digest

    actual = _walk_files(model_root)
    if set(actual) != set(expected):
        raise ModelAssetError("model file set does not match manifest")
    total = 0
    for relative, path in actual.items():
        try:
            validate_regular_file(path)
            before = path.stat(follow_symlinks=False)
            payload = path.read_bytes()
            validate_regular_file(path)
            after = path.stat(follow_symlinks=False)
        except (OSError, FileSafetyError) as error:
            raise ModelAssetError(f"unsafe model file: {path}") from error
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_nlink)
            != (after.st_dev, after.st_ino, after.st_size, after.st_nlink)
            or after.st_size != len(payload)
        ):
            raise ModelAssetError(f"model file identity changed: {relative}")
        size, digest = expected[relative]
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ModelAssetError(f"model file content mismatch: {relative}")
        total += len(payload)
    if manifest["file_count"] != len(actual) or manifest["total_size_bytes"] != total:
        raise ModelAssetError("model manifest totals do not match files")
    return VerifiedModelAssets(model_root.resolve(), len(actual), total)
