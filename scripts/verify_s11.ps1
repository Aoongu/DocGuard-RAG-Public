[CmdletBinding()]
param([string]$RerankerModelRoot = "")

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) { $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache" }
$PreviousLocation = Get-Location

function Assert-LastExitCode { param([Parameter(Mandatory = $true)][string]$Step) if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" } }

try {
    Set-Location -LiteralPath $ProjectRoot
    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status"
    if ($InitialStatus.Count -ne 0) { throw "Worktree must be clean before S11 verification" }

    Write-Output "[S11] Run S10 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "scripts\verify_s10.ps1") -RerankerModelRoot $RerankerModelRoot
    Assert-LastExitCode "S10 gate"

    Write-Output "[S11] Run engineering tests"
    $S11Tests = @(Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\unit") -Filter "test_s11_*.py" -File | Sort-Object FullName | ForEach-Object { $_.FullName })
    $S11Tests += @(Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\integration") -Filter "test_s11_*.py" -File | Sort-Object FullName | ForEach-Object { $_.FullName })
    if ($S11Tests.Count -eq 0) { throw "No S11 tests were found" }
    & uv run pytest -q @S11Tests
    Assert-LastExitCode "S11 tests"

    Write-Output "[S11] Run deterministic fixture regression"
    & uv run python -m docguard.engineering.regression --fixture data/eval/s11-regression-v1.json --config configs/engineering/s11-engineering-v1.json
    Assert-LastExitCode "S11 fixture regression"

    $Manifest = Get-Content -LiteralPath (Join-Path $ProjectRoot "data\eval\s4-golden-v1\manifest.json") -Raw | ConvertFrom-Json
    if ($Manifest.test_execution_status -ne "locked_until_s12") { throw "S11 must preserve locked_until_s12" }

    & uv run ruff check src tests
    Assert-LastExitCode "Ruff"

    if (Get-Command docker -ErrorAction SilentlyContinue) {
        & docker compose config --quiet
        Assert-LastExitCode "Docker Compose config"
        Write-Output "S11_DOCKER_CONFIG_PASS"
    }
    else {
        Write-Output "S11_DOCKER_RUNTIME_SKIPPED: docker command is unavailable; CI must run image and service checks"
    }

    & git diff --check
    Assert-LastExitCode "git diff --check"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "final git status"
    if (Compare-Object $InitialStatus $FinalStatus) { throw "S11 verification changed the tracked worktree" }
    Write-Output "S11_VERIFY_PASS"
}
catch { Write-Error "S11_VERIFY_FAIL: $($_.Exception.Message)"; exit 1 }
finally { Set-Location -LiteralPath $PreviousLocation }
