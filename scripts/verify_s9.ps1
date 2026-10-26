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
        throw "Worktree must be clean before S9 verification"
    }

    Write-Output "[S9] Run S8 offline gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_s8.ps1") `
        -RerankerModelRoot $RerankerModelRoot
    Assert-LastExitCode "S8 gate"

    Write-Output "[S9] Verify API contract and bootstrap"
    $S9UnitTests = @(
        Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\unit") `
            -File -Filter "test_s9_*.py" |
            Sort-Object FullName |
            ForEach-Object { $_.FullName }
    )
    $S9IntegrationTests = @(
        Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\integration") `
            -File -Filter "test_s9_*.py" |
            Sort-Object FullName |
            ForEach-Object { $_.FullName }
    )
    & uv run pytest @S9UnitTests @S9IntegrationTests -q
    Assert-LastExitCode "S9 tests"

    & uv run python -c `
        "from docguard.api.main import app; paths=set(app.openapi()['paths']); required={'/api/v1/health','/api/v1/corpora','/api/v1/corpora/rebuild','/api/v1/query','/api/v1/query/stream','/api/v1/evidence/{evidence_id}','/api/v1/evaluations','/api/v1/evaluations/{run_id}'}; assert required <= paths"
    Assert-LastExitCode "OpenAPI route verification"

    $ErrorActionPreference = "Continue"
    try {
        $TestLockText = (& uv run python -c `
            "from docguard.api.main import EVALUATIONS; import asyncio; asyncio.run(EVALUATIONS.create_job(method='m4', split='test'))" `
            2>&1 | Out-String)
        $TestLockExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = "Stop"
    }
    if ($TestLockExitCode -eq 0 -or $TestLockText -notmatch "S12") {
        throw "S9 Test split unexpectedly created an evaluation job"
    }

    Write-Output "[S9] Run full regression, lint, and package build"
    & uv run pytest -q
    Assert-LastExitCode "full tests"
    & uv run ruff check src tests scripts
    Assert-LastExitCode "Ruff"
    & git diff --check
    Assert-LastExitCode "git diff --check"

    $BuildRoot = Join-Path $ProjectRoot "reports\generated\s9-package-build"
    if (Test-Path -LiteralPath $BuildRoot) {
        Remove-Item -LiteralPath $BuildRoot -Recurse -Force
    }
    & uv build --out-dir $BuildRoot
    Assert-LastExitCode "package build"
    $Wheels = @(Get-ChildItem -LiteralPath $BuildRoot -File -Filter "*.whl")
    $SourceDistributions = @(
        Get-ChildItem -LiteralPath $BuildRoot -File -Filter "*.tar.gz"
    )
    if ($Wheels.Count -ne 1 -or $SourceDistributions.Count -ne 1) {
        throw "Expected one wheel and one source distribution"
    }

    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "final git status"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S9 verification changed the tracked worktree"
    }
    Write-Output "S9_VERIFY_PASS"
}
catch {
    Write-Error "S9_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
}
