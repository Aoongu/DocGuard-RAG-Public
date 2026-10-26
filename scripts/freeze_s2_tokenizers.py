"""Freeze the two approved S2 tokenizers without downloading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen

from tokenizers import Tokenizer

_HUGGING_FACE = "https://huggingface.co"
_MAX_TOKENIZER_BYTES = 32 * 1024 * 1024
_MAX_METADATA_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ModelPin:
    name: str
    revision: str


MODELS = (
    ModelPin(
        "intfloat/multilingual-e5-small",
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    ),
    ModelPin(
        "BAAI/bge-reranker-base",
        "2cfc18c9415c912f9d8155881c133215df768a70",
    ),
)

class ResponseOpener(Protocol):
    def __call__(
        self, request: Request, *, timeout: float
    ) -> AbstractContextManager[BinaryIO]: ...


def _download(opener: ResponseOpener, url: str, maximum: int) -> bytes:
    request = Request(url, headers={"User-Agent": "DocGuard-RAG-tokenizer-freezer/1"})
    with opener(request, timeout=60.0) as response:
        payload = response.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise ValueError(f"download has invalid size: {url}")
    return payload


def _atomic_write(
    path: Path, payload: bytes, *, replace_existing: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"asset target is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        if not replace_existing:
            raise ValueError(f"existing content-addressed asset differs: {path}")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _license_from_metadata(metadata: bytes, readme: bytes) -> str:
    parsed: Any = json.loads(metadata)
    license_value = None
    if isinstance(parsed, dict):
        license_value = parsed.get("license")
        card_data = parsed.get("cardData")
        if license_value is None and isinstance(card_data, dict):
            license_value = card_data.get("license")
    if isinstance(license_value, str) and license_value.lower() == "mit":
        return "MIT"
    if b"license: mit" in readme.lower():
        return "MIT"
    raise ValueError("model metadata does not declare the required MIT license")


def freeze_tokenizers(
    repository_root: Path,
    *,
    opener: ResponseOpener = urlopen,
    retrieved_at: str | None = None,
) -> Path:
    """Download, validate and content-address the approved tokenizer files."""
    root = repository_root.resolve()
    output_root = root / "data" / "tokenizers"
    timestamp = retrieved_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries: list[dict[str, object]] = []
    for model in MODELS:
        encoded_name = quote(model.name, safe="/")
        revision = quote(model.revision, safe="")
        tokenizer_url = (
            f"{_HUGGING_FACE}/{encoded_name}/resolve/{revision}/tokenizer.json"
        )
        readme_url = f"{_HUGGING_FACE}/{encoded_name}/resolve/{revision}/README.md"
        metadata_url = (
            f"{_HUGGING_FACE}/api/models/{encoded_name}/revision/{revision}"
        )
        tokenizer = _download(opener, tokenizer_url, _MAX_TOKENIZER_BYTES)
        readme = _download(opener, readme_url, _MAX_METADATA_BYTES)
        metadata = _download(opener, metadata_url, _MAX_METADATA_BYTES)
        Tokenizer.from_str(tokenizer.decode("utf-8"))
        digest = hashlib.sha256(tokenizer).hexdigest()
        relative_asset = Path("data", "tokenizers", digest, "tokenizer.json")
        _atomic_write(root / relative_asset, tokenizer)
        entries.append(
            {
                "name": model.name,
                "revision": model.revision,
                "tokenizer_url": tokenizer_url,
                "readme_url": readme_url,
                "metadata_url": metadata_url,
                "license": _license_from_metadata(metadata, readme),
                "asset_path": relative_asset.as_posix(),
                "size_bytes": len(tokenizer),
                "sha256": digest,
                "readme_size_bytes": len(readme),
                "readme_sha256": hashlib.sha256(readme).hexdigest(),
                "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "retrieved_at": timestamp,
        "models": entries,
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_path = output_root / "manifest.json"
    _atomic_write(manifest_path, manifest_payload, replace_existing=True)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    manifest = freeze_tokenizers(args.repository_root)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
