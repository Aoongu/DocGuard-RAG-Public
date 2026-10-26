import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Self

from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "freeze_s2_tokenizers.py"
SPEC = importlib.util.spec_from_file_location("freeze_s2_tokenizers", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
MODELS = MODULE.MODELS
freeze_tokenizers = MODULE.freeze_tokenizers


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]


def _minimal_tokenizer() -> bytes:
    return json.dumps(
        {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1},
                "unk_token": "[UNK]",
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_freezer_uses_only_fixed_revision_metadata_and_tokenizer_urls(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    tokenizer = _minimal_tokenizer()

    def opener(request: Any, *, timeout: float) -> _Response:
        url = request.full_url
        calls.append(url)
        assert timeout == 60.0
        if url.endswith("/tokenizer.json"):
            return _Response(tokenizer)
        if url.endswith("/README.md"):
            return _Response(b"---\nlicense: mit\n---\n# Model card\n")
        if "/api/models/" in url:
            return _Response(json.dumps({"license": "mit"}).encode("utf-8"))
        raise AssertionError(f"unexpected URL: {url}")

    manifest_path = freeze_tokenizers(
        repository_root=tmp_path,
        opener=opener,
        retrieved_at="2026-10-14T00:00:00Z",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["retrieved_at"] == "2026-10-14T00:00:00Z"
    assert [entry["name"] for entry in manifest["models"]] == [
        model.name for model in MODELS
    ]
    assert len(calls) == 6
    for model in MODELS:
        assert sum(model.revision in url for url in calls) == 3
    assert not any(
        url.endswith((".bin", ".safetensors", ".pt", ".onnx")) for url in calls
    )


def test_freezer_content_addresses_valid_tokenizer_and_deduplicates_assets(
    tmp_path: Path,
) -> None:
    tokenizer = _minimal_tokenizer()

    def opener(request: Any, *, timeout: float) -> _Response:
        if request.full_url.endswith("/tokenizer.json"):
            return _Response(tokenizer)
        if request.full_url.endswith("/README.md"):
            return _Response(b"license: mit")
        return _Response(json.dumps({"license": "mit"}).encode("utf-8"))

    manifest_path = freeze_tokenizers(
        repository_root=tmp_path,
        opener=opener,
        retrieved_at="2026-10-14T00:00:00Z",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(tokenizer).hexdigest()
    asset = tmp_path / "data" / "tokenizers" / digest / "tokenizer.json"

    assert asset.read_bytes() == tokenizer
    assert list((tmp_path / "data" / "tokenizers").glob("*/tokenizer.json")) == [
        asset
    ]
    for entry in manifest["models"]:
        assert entry["asset_path"] == f"data/tokenizers/{digest}/tokenizer.json"
        assert entry["size_bytes"] == len(tokenizer)
        assert entry["sha256"] == digest
        assert entry["license"] == "MIT"
        assert entry["readme_sha256"] == hashlib.sha256(b"license: mit").hexdigest()


def test_freezer_atomically_refreshes_manifest_without_rewriting_assets(
    tmp_path: Path,
) -> None:
    tokenizer = _minimal_tokenizer()

    def opener(request: Any, *, timeout: float) -> _Response:
        if request.full_url.endswith("/tokenizer.json"):
            return _Response(tokenizer)
        if request.full_url.endswith("/README.md"):
            return _Response(b"license: mit")
        return _Response(b'{"license":"mit"}')

    first = freeze_tokenizers(
        repository_root=tmp_path,
        opener=opener,
        retrieved_at="2026-10-14T00:00:00Z",
    )
    asset = next((tmp_path / "data" / "tokenizers").glob("*/tokenizer.json"))
    first_asset_mtime = asset.stat().st_mtime_ns
    second = freeze_tokenizers(
        repository_root=tmp_path,
        opener=opener,
        retrieved_at="2026-10-14T01:00:00Z",
    )

    assert second == first
    assert json.loads(second.read_text(encoding="utf-8"))["retrieved_at"] == (
        "2026-10-14T01:00:00Z"
    )
    assert asset.stat().st_mtime_ns == first_asset_mtime


def test_committed_manifest_assets_are_hash_pinned_and_offline_loadable() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "data" / "tokenizers" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == 1
    assert [entry["name"] for entry in manifest["models"]] == [
        model.name for model in MODELS
    ]
    for model, entry in zip(MODELS, manifest["models"], strict=True):
        asset = PROJECT_ROOT / entry["asset_path"]
        payload = asset.read_bytes()
        assert entry["revision"] == model.revision
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["license"] == "MIT"
        Tokenizer.from_str(payload.decode("utf-8"))

