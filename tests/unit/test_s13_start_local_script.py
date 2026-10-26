from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_start_local_script_runs_verified_api_and_frontend_workflow() -> None:
    script = (ROOT / "scripts/start_local.ps1").read_text(encoding="utf-8")

    for command in (
        "uv sync --frozen --group dev",
        "npm ci",
        "npm test -- --run",
        "npm run build",
        "npm run dev -- --host 127.0.0.1",
        "uv run docguard-api",
    ):
        assert command in script
    assert "http://127.0.0.1:5173" in script
    assert "Start-Process" in script
    assert "DEEPSEEK_API_KEY" in script
    assert "EnvironmentVariableTarget]::User" in script


def test_start_local_script_keeps_helpers_bounded_and_checks_exit_codes() -> None:
    script = (ROOT / "scripts/start_local.ps1").read_text(encoding="utf-8")

    assert "$ErrorActionPreference = \"Stop\"" in script
    assert "Assert-LastExitCode" in script
    assert "AddMinutes(2)" in script
    assert "-WindowStyle Hidden" in script
