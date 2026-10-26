from __future__ import annotations

import json
from pathlib import Path

import pytest

from docguard.evaluation.final_protocol import (
    load_final_protocol,
    verify_final_protocol_inputs,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/evaluation/s12-final-v1.json"


def _payload() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_repository_final_protocol_is_complete_and_compatible() -> None:
    protocol = load_final_protocol(CONFIG)

    assert protocol.protocol_version == "s12-final-v1"
    assert protocol.dataset.test_sha256 == (
        "67bbafdb8841a15f58de0cd66cf14f488f142a2f1e44bff360ef17647e56adeb"
    )
    assert protocol.method_order == (
        "b0",
        "b1",
        "m1",
        "m2",
        "b2",
        "m3",
        "m4",
        "m4_no_version_filter",
        "m4_no_citation_validation",
    )
    assert protocol.ablations == {
        "no_bm25": "b1",
        "no_dense": "b0",
        "no_cross_encoder": "m1",
        "prompt_only": "m3",
        "no_version_filter": "m4_no_version_filter",
        "no_citation_validation": "m4_no_citation_validation",
    }
    assert protocol.max_retries == 0
    assert protocol.existing_record_policy == "skip_any_record"
    verify_final_protocol_inputs(ROOT, protocol)


def test_final_protocol_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = _payload()
    payload["unknown"] = True
    path = tmp_path / "s12.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="fields"):
        load_final_protocol(path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["dataset"].update(test_sha256="0" * 64), "Test SHA-256"),
        (lambda value: value.update(max_retries=1), "max_retries"),
        (
            lambda value: value.update(existing_record_policy="retry_failures"),
            "existing_record_policy",
        ),
        (
            lambda value: value["methods"]["m1"]["parameters"].update(rrf_k=60),
            "M1 parameters",
        ),
        (
            lambda value: value["methods"]["m2"]["parameters"].update(
                rerank_depth=20
            ),
            "M2 parameters",
        ),
        (
            lambda value: value["ablations"].update(no_dense="b1"),
            "ablations",
        ),
    ],
)
def test_final_protocol_rejects_drift(
    tmp_path: Path, mutator: object, message: str
) -> None:
    payload = _payload()
    mutator(payload)  # type: ignore[operator]
    path = tmp_path / "s12.json"
    _write(path, payload)

    with pytest.raises(ValueError, match=message):
        load_final_protocol(path)


def test_final_protocol_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    payload = _payload()
    payload["methods"]["b1"]["run_id"] = payload["methods"]["b0"]["run_id"]
    path = tmp_path / "s12.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="run ID"):
        load_final_protocol(path)


def test_final_protocol_input_verification_fails_closed_on_hash_drift(
    tmp_path: Path,
) -> None:
    protocol = load_final_protocol(CONFIG)
    root = tmp_path / "repo"
    for item in protocol.frozen_inputs:
        target = root / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / item.path).read_bytes())
    manifest = root / protocol.dataset.manifest_path
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes((ROOT / protocol.dataset.manifest_path).read_bytes())
    (root / protocol.dataset.test_path).parent.mkdir(parents=True, exist_ok=True)
    (root / protocol.dataset.test_path).write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="test dataset hash"):
        verify_final_protocol_inputs(root, protocol)
