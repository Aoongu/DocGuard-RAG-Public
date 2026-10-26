from __future__ import annotations

from copy import deepcopy

import pytest

from docguard.evaluation.preflight import S12PreflightError, validate_preflight_state


def _state() -> dict[str, object]:
    return {
        "git_clean": True,
        "credential_present": True,
        "old_seal_present": False,
        "existing_output_dirs": [],
        "dataset": {
            "status": "frozen",
            "human_review_complete": True,
            "test_double_review_complete": True,
            "test_execution_status": "locked_until_s12",
            "test_review_count": 60,
        },
        "assets": {
            "chunks": True,
            "dense_index": True,
            "bm25_index": True,
            "embedding_model": True,
            "reranker_model": True,
            "tokenizers": True,
        },
    }


def test_valid_preflight_state_passes() -> None:
    validate_preflight_state(_state())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("git_clean", False, "working tree"),
        ("credential_present", False, "credential"),
        ("old_seal_present", True, "seal"),
        ("existing_output_dirs", ["m4"], "output"),
    ],
)
def test_preflight_rejects_unsafe_execution_state(
    field: str, value: object, message: str
) -> None:
    state = _state()
    state[field] = value

    with pytest.raises(S12PreflightError, match=message):
        validate_preflight_state(state)


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "human_review_complete",
        "test_double_review_complete",
        "test_review_count",
    ],
)
def test_preflight_rejects_incomplete_test_review(field: str) -> None:
    state = _state()
    dataset = deepcopy(state["dataset"])
    assert isinstance(dataset, dict)
    dataset[field] = False if field != "status" else "candidate"
    state["dataset"] = dataset

    with pytest.raises(S12PreflightError, match="dataset review"):
        validate_preflight_state(state)


def test_preflight_rejects_missing_or_mismatched_asset() -> None:
    state = _state()
    assets = deepcopy(state["assets"])
    assert isinstance(assets, dict)
    assets["reranker_model"] = False
    state["assets"] = assets

    with pytest.raises(S12PreflightError, match="asset verification"):
        validate_preflight_state(state)
