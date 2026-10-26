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
$TempRoot = $null
$TempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/')

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

try {
    Set-Location -LiteralPath $ProjectRoot
    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status --short"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before S7 verification:`n$($InitialStatus -join [Environment]::NewLine)"
    }

    Write-Output "[S7] Run S6 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_s6.ps1") `
        -RerankerModelRoot $RerankerModelRoot
    Assert-LastExitCode "S6 gate"

    Write-Output "[S7] Validate frozen generation config, schema, and prompt"
    & uv run python -m docguard.evaluation.generation_cli validate-config `
        --generation-config (Join-Path $ProjectRoot "configs\generation\s7-generation-v1.json")
    Assert-LastExitCode "S7 frozen generation config"

    Write-Output "[S7] Verify two independent Mock smokes and results_sha256 comparison"
    & uv run pytest tests\integration\test_s7_mock_generation.py -q
    Assert-LastExitCode "S7 independent Mock smoke comparison"

    $Manifest = Get-Content -LiteralPath `
        (Join-Path $ProjectRoot "data\eval\s4-golden-v1\manifest.json") -Raw |
        ConvertFrom-Json
    if ($Manifest.test_execution_status -ne "locked_until_s12") {
        throw "S7 requires locked_until_s12 Test status"
    }
    $TempRoot = Join-Path $TempBase ("docguard-s7-verify-" + [Guid]::NewGuid().ToString("N"))
    $LockedOutput = Join-Path $TempRoot "forbidden-s7-test"
    New-Item -ItemType Directory -Path $TempRoot | Out-Null
    $ErrorActionPreference = "Continue"
    try {
        $LockedText = (& uv run python -m docguard.evaluation.generation_cli run-b2 `
            --dataset (Join-Path $ProjectRoot "data\eval\s4-golden-v1\test.jsonl") `
            --output $LockedOutput --run-id forbidden-s7-test 2>&1 | Out-String)
        $LockedExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = "Stop"
    }
    if ($LockedExitCode -eq 0 -or $LockedText -notmatch "S12" -or
        (Test-Path -LiteralPath $LockedOutput)) {
        throw "S7 Test split unexpectedly executed"
    }

    Write-Output "[S7] Run S7 unit tests"
    $S7UnitTests = @(
        Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\unit") `
            -File -Filter "test_s7_*.py" |
            Sort-Object FullName |
            ForEach-Object { $_.FullName }
    )
    if ($S7UnitTests.Count -eq 0) {
        throw "No S7 unit tests were discovered"
    }
    & uv run pytest @S7UnitTests -q
    Assert-LastExitCode "S7 tests"

    Write-Output "[S7] Run full regression suite"
    & uv run pytest -q
    Assert-LastExitCode "full tests"
    & uv run ruff check src tests scripts
    Assert-LastExitCode "Ruff"
    & git diff --check
    Assert-LastExitCode "git diff --check"

    Write-Output "[S7] Verify clean worktree"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "git status --short"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S7 verification changed the worktree"
    }
    Write-Output "S7_MOCK_GATE_PASS"
}
catch {
    Write-Error "S7_MOCK_GATE_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
    if ($null -ne $TempRoot -and (Test-Path -LiteralPath $TempRoot)) {
        $SafeTempRoot = [IO.Path]::GetFullPath($TempRoot)
        $ExpectedPrefix = $TempBase + [IO.Path]::DirectorySeparatorChar
        $Leaf = [IO.Path]::GetFileName($SafeTempRoot)
        if (-not $SafeTempRoot.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not $Leaf.StartsWith("docguard-s7-verify-", [StringComparison]::Ordinal)) {
            throw "Refusing to remove unexpected temporary path: $SafeTempRoot"
        }
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
        if (Test-Path -LiteralPath $TempRoot) {
            throw "Failed to remove S7 verification directory: $TempRoot"
        }
    }
}
