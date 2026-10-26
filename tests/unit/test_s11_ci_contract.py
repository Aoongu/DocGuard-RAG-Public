from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_ci_runs_python_frontend_regression_and_qdrant_parity() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for required in (
        "uv sync --frozen --group dev",
        "uv run pytest -q",
        "uv run ruff check src tests",
        "npm ci",
        "npm test -- --run",
        "npm run typecheck",
        "npm run build",
        "docguard.engineering.regression",
        "S11_QDRANT_URL: http://127.0.0.1:6333",
        "test_s11_qdrant_parity.py",
        "docker compose config",
        "docker compose up -d qdrant api",
        "docker compose logs api",
        "docker compose down",
        "s11_ci_persistence",
        "collections/s11_ci_persistence/points/1",
    ):
        assert required in workflow
    assert "test_execution_status" in workflow
    assert "locked_until_s12" in workflow


def test_s11_gate_preserves_test_lock_and_checks_container_contract() -> None:
    gate = (ROOT / "scripts/verify_s11.ps1").read_text(encoding="utf-8")
    assert "verify_s10.ps1" in gate
    assert "test_s11_" in gate
    assert "locked_until_s12" in gate
    assert "docguard.engineering.regression" in gate
    assert "S11_VERIFY_PASS" in gate
