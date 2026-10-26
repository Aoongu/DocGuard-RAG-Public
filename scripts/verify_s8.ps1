[CmdletBinding()]
param(
    [string]$RerankerModelRoot = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
}
if ([string]::IsNullOrWhiteSpace($RerankerModelRoot)) {
    $RerankerModelRoot = Join-Path $ProjectRoot "data\models\s6-bge-reranker-base"
}
$PreviousLocation = Get-Location

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

try {
    Set-Location -LiteralPath $ProjectRoot
    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before S8 verification"
    }

    Write-Output "[S8] Run S7 offline gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_s7.ps1") `
        -RerankerModelRoot $RerankerModelRoot
    Assert-LastExitCode "S7 gate"

    Write-Output "[S8] Verify frozen reports and manifest"
    & uv run pytest tests\integration\test_s8_refusal_calibration.py -q
    Assert-LastExitCode "S8 report verification"
    & uv run python -m docguard.evaluation.refusal_cli summarize `
        --output reports\s8_refusal\validation_selected
    Assert-LastExitCode "Validation summary verification"
    & uv run python -m docguard.evaluation.refusal_cli summarize `
        --output reports\s8_refusal\development_diagnostic
    Assert-LastExitCode "Development summary verification"

    Write-Output "[S8] Verify Test lock"
    $ErrorActionPreference = "Continue"
    try {
        $LockedText = (& uv run python -m docguard.evaluation.refusal_cli calibrate `
            --split test --dataset missing --results missing --chunks missing `
            --chunk-manifest missing --output forbidden-s8-test 2>&1 | Out-String)
        $LockedExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = "Stop"
    }
    if ($LockedExitCode -eq 0 -or $LockedText -notmatch "S12" -or
        (Test-Path -LiteralPath (Join-Path $ProjectRoot "forbidden-s8-test"))) {
        throw "S8 Test split unexpectedly executed"
    }

    Write-Output "[S8] Run S8 tests and full regression"
    $S8UnitTests = @(
        Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\unit") `
            -File -Filter "test_s8_*.py" |
            Sort-Object FullName |
            ForEach-Object { $_.FullName }
    )
    & uv run pytest @S8UnitTests -q
    Assert-LastExitCode "S8 unit tests"
    & uv run pytest -q
    Assert-LastExitCode "full tests"
    & uv run ruff check src tests scripts
    Assert-LastExitCode "Ruff"
    & git diff --check
    Assert-LastExitCode "git diff --check"

    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "final git status"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S8 verification changed the worktree"
    }
    Write-Output "S8_VERIFY_PASS"
}
catch {
    Write-Error "S8_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
}
