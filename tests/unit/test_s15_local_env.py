import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("powershell") is None, reason="Windows PowerShell is required"
)

ROOT = Path(__file__).resolve().parents[2]


def run_import(path, body):
    helper = (ROOT / "scripts/project_env.ps1").as_posix().replace("'", "''")
    config = path.as_posix().replace("'", "''")
    environment = os.environ.copy()
    environment["DEEPSEEK_API_KEY"] = "existing-model-fixture"
    script = (
        ". '" + helper + "'; Import-DocGuardEnvironment -Path '" + config + "'; " + body
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )
    return json.loads(result.stdout)


def test_env_file_loads_literals_keeps_existing_model_and_overrides_new_key(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        '# comment\nDEEPSEEK_API_KEY=\nSERPAPI_API_KEY="search-fixture"\nDOCGUARD_SEARCH_PROVIDER=serpapi\n',
        encoding="utf-8",
    )
    values = run_import(
        path,
        "@{model=$env:DEEPSEEK_API_KEY;search=$env:SERPAPI_API_KEY;provider=$env:DOCGUARD_SEARCH_PROVIDER} | ConvertTo-Json -Compress",
    )
    assert values == {
        "model": "existing-model-fixture",
        "search": "search-fixture",
        "provider": "serpapi",
    }


def test_env_file_never_evaluates_code_or_accepts_unknown_environment_keys(tmp_path):
    path = tmp_path / ".env"
    marker = (tmp_path / "executed").as_posix()
    expression = "$(New-Item -ItemType File -Path '" + marker + "')"
    path.write_text(
        "SERPAPI_API_KEY=" + expression + "\nDOCGUARD_UNKNOWN_SECRET=not-allowed\n",
        encoding="utf-8",
    )
    values = run_import(
        path,
        "@{search=$env:SERPAPI_API_KEY;unknown=$env:DOCGUARD_UNKNOWN_SECRET} | ConvertTo-Json -Compress",
    )
    assert values["search"] == expression
    assert not values["unknown"]
    assert not (tmp_path / "executed").exists()
