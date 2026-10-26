from pathlib import Path

from docguard.evaluation.final_report import verify_final_report

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_s12_delivery_is_internally_consistent() -> None:
    verified = verify_final_report(ROOT)

    assert verified == {
        "method_count": 9,
        "result_row_count": 540,
        "reviewed_count": 9,
        "figure_count": 4,
    }


def test_s12_gate_is_read_only_and_chains_prior_gate() -> None:
    gate = (ROOT / "scripts" / "verify_s12.ps1").read_text(encoding="utf-8")

    required_fragments = (
        "verify_s11.ps1",
        "--verify-only",
        "test_s12_*.py",
        "uv run pytest -q",
        "uv run ruff check",
        "npm ci",
        "npm run test",
        "npm run typecheck",
        "npm run build",
        "S12_VERIFY_PASS",
    )
    for fragment in required_fragments:
        assert fragment in gate
    assert "run_s12.ps1" not in gate
    assert "DEEPSEEK_API_KEY" not in gate
