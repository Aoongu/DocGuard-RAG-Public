from __future__ import annotations

from pathlib import Path

import pytest

from docguard.evaluation.final_cli import _metadata, _parser
from docguard.evaluation.final_protocol import load_final_protocol

ROOT = Path(__file__).parents[2]
PROTOCOL = load_final_protocol(ROOT / "configs/evaluation/s12-final-v1.json")


def test_cli_accepts_only_registered_s12_routes_without_output_override() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "run",
            "--method",
            "m4_no_citation_validation",
            "--project-root",
            str(ROOT),
            "--asset-root",
            str(ROOT),
        ]
    )

    assert args.method == "m4_no_citation_validation"
    assert parser.parse_args(["run", "--method", "b0"]).method == "b0"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--method", "unknown"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--method", "m4", "--output", "elsewhere"])


def test_powershell_wrapper_has_fixed_order_and_no_retry_loop() -> None:
    script = (ROOT / "scripts/run_s12.ps1").read_text(encoding="utf-8")

    assert "@('b0', 'b1', 'm1', 'm2')" in script
    assert "'b2'," in script
    assert "'m4_no_citation_validation'" in script
    assert "GetEnvironmentVariable('DEEPSEEK_API_KEY', 'User')" in script
    assert "for ($" not in script.casefold()
    assert "Start-Sleep" not in script


def test_cli_metadata_is_derived_from_frozen_protocol() -> None:
    metadata = _metadata(ROOT, PROTOCOL, "m3", git_commit="abc123")

    assert metadata["run_id"] == "s12-test-m3-v1"
    assert metadata["dataset_sha256"] == PROTOCOL.dataset.test_sha256
    assert metadata["parameters"] == PROTOCOL.methods["m3"].parameters
    assert metadata["git_commit"] == "abc123"
