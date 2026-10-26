"""Offline token counting backed by hash-pinned tokenizer assets."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tokenizers import Tokenizer

_MAX_CHUNK_TOKENS = 384
_SHA256_LENGTH = 64


class TokenBudgetError(ValueError):
    """Raised when tokenizer provenance or a token limit is invalid."""


def _safe_asset_path(repository_root: Path, value: object) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise TokenBudgetError("asset_path must be a relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative == PurePosixPath("."):
        raise TokenBudgetError("asset_path must stay inside repository_root")
    target = repository_root.joinpath(*relative.parts)
    try:
        target.resolve(strict=False).relative_to(repository_root)
    except ValueError as error:
        raise TokenBudgetError("asset_path escapes repository_root") from error
    return target


def _read_regular_single_link(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise TokenBudgetError(f"tokenizer asset is unavailable: {path}") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or bool(attributes & reparse)
    ):
        raise TokenBudgetError(f"tokenizer asset must be a regular file: {path}")
    if metadata.st_nlink != 1:
        raise TokenBudgetError(f"tokenizer asset must not be a hardlink: {path}")
    try:
        with path.open("rb") as file:
            opened = os.fstat(file.fileno())
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise TokenBudgetError(f"tokenizer asset changed before open: {path}")
            payload = file.read()
            after = os.fstat(file.fileno())
    except OSError as error:
        raise TokenBudgetError(f"unable to read tokenizer asset: {path}") from error
    if (
        (after.st_dev, after.st_ino, after.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or len(payload) != opened.st_size
    ):
        raise TokenBudgetError(f"tokenizer asset changed while reading: {path}")
    return payload


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Count against every frozen tokenizer and enforce their shared limit."""

    _tokenizers: tuple[Tokenizer, ...]

    @classmethod
    def from_manifest(
        cls, manifest_path: Path, repository_root: Path
    ) -> TokenBudget:
        root = repository_root.resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TokenBudgetError("unable to load tokenizer manifest") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise TokenBudgetError("tokenizer manifest schema_version must be 1")
        entries = manifest.get("models")
        if not isinstance(entries, list) or len(entries) != 2:
            raise TokenBudgetError("tokenizer manifest must contain exactly two models")
        loaded: list[Tokenizer] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise TokenBudgetError("tokenizer model entry must be an object")
            target = _safe_asset_path(root, entry.get("asset_path"))
            payload = _read_regular_single_link(target)
            expected_size = entry.get("size_bytes")
            expected_hash = entry.get("sha256")
            if type(expected_size) is not int or expected_size != len(payload):
                raise TokenBudgetError(f"tokenizer asset size mismatch: {target}")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != _SHA256_LENGTH
                or hashlib.sha256(payload).hexdigest() != expected_hash
            ):
                raise TokenBudgetError(f"tokenizer asset SHA-256 mismatch: {target}")
            try:
                tokenizer = Tokenizer.from_str(payload.decode("utf-8"))
            except (UnicodeError, ValueError) as error:
                raise TokenBudgetError(f"invalid tokenizer asset: {target}") from error
            tokenizer.no_padding()
            tokenizer.no_truncation()
            loaded.append(tokenizer)
        return cls(tuple(loaded))

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return max(
            len(tokenizer.encode(text, add_special_tokens=True).ids)
            for tokenizer in self._tokenizers
        )

    def validate_chunk(self, text: str) -> int:
        count = self.count(text)
        if count > _MAX_CHUNK_TOKENS:
            raise TokenBudgetError(
                f"chunk has {count} tokens, exceeding {_MAX_CHUNK_TOKENS}"
            )
        return count
