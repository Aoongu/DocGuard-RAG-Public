[CmdletBinding()]
param([string]$RerankerModelRoot = "")

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
}
$PreviousLocation = Get-Location

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$Step)
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" }
}

try {
    Set-Location -LiteralPath $ProjectRoot
    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before S12 verification"
    }

    Write-Output "[S12] Run S11 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_s11.ps1") `
        -RerankerModelRoot $RerankerModelRoot
    Assert-LastExitCode "S11 gate"

    Write-Output "[S12] Verify frozen raw results, ledger, review and figures"
    & uv run python -m docguard.evaluation.final_report `
        --repo-root $ProjectRoot --verify-only
    Assert-LastExitCode "S12 report verification"

    Write-Output "[S12] Run S12 contract tests"
    $S12Tests = @(Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\unit") `
        -Filter "test_s12_*.py" -File | Sort-Object FullName | ForEach-Object { $_.FullName })
    if ($S12Tests.Count -eq 0) { throw "No S12 tests were found" }
    & uv run pytest -q @S12Tests
    Assert-LastExitCode "S12 tests"

    Write-Output "[S12] Run full Python regression"
    & uv run pytest -q
    Assert-LastExitCode "full pytest"
    & uv run ruff check src tests scripts/render_s12_figures.py
    Assert-LastExitCode "Ruff"

    Write-Output "[S12] Run frontend reproducibility gate"
    Set-Location -LiteralPath (Join-Path $ProjectRoot "frontend")
    & npm ci
    Assert-LastExitCode "npm ci"
    & npm run test -- --run
    Assert-LastExitCode "frontend tests"
    & npm run typecheck
    Assert-LastExitCode "frontend typecheck"
    & npm run build
    Assert-LastExitCode "frontend build"
    Set-Location -LiteralPath $ProjectRoot

    & git diff --check
    Assert-LastExitCode "git diff --check"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "final git status"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S12 verification changed the tracked worktree"
    }
    Write-Output "S12_VERIFY_PASS"
}
catch {
    Write-Error "S12_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
}
