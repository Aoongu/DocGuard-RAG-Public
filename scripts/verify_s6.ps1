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

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-SavedRunContracts {
    param([Parameter(Mandatory = $true)][string]$ReportsRoot)

    $RunFiles = @(Get-ChildItem -LiteralPath $ReportsRoot -Recurse -File -Filter "run.json")
    $SummaryFiles = @(
        Get-ChildItem -LiteralPath $ReportsRoot -Recurse -File -Filter "summary.json"
    )
    if ($RunFiles.Count -ne 4 -or $SummaryFiles.Count -ne 4) {
        throw "S6 expected four saved runs"
    }
    foreach ($RunFile in $RunFiles) {
        $Run = Get-Content -LiteralPath $RunFile.FullName -Raw | ConvertFrom-Json
        if ($Run.split -eq "test" -or $Run.method -ne "m2" -or
            $Run.parameters.rerank_enabled -ne $true -or
            $Run.parameters.rerank_depth -notin @(10, 20, 50) -or
            $Run.reranker_model_name -ne "BAAI/bge-reranker-base" -or
            $Run.reranker_model_revision -ne "2cfc18c9415c912f9d8155881c133215df768a70" -or
            $Run.reranker_max_length -ne 512) {
            throw "S6 saved run contract mismatch: $($RunFile.FullName)"
        }
        $ResultPath = Join-Path $RunFile.DirectoryName "results.jsonl"
        $FailurePath = Join-Path $RunFile.DirectoryName "failures.jsonl"
        if (@(Get-Content -LiteralPath $ResultPath).Count -ne 60 -or
            @(Get-Content -LiteralPath $FailurePath).Count -ne 0) {
            throw "S6 saved run row/failure count mismatch: $($RunFile.DirectoryName)"
        }
    }
    foreach ($SummaryFile in $SummaryFiles) {
        $Summary = Get-Content -LiteralPath $SummaryFile.FullName -Raw | ConvertFrom-Json
        $ResultPath = Join-Path $SummaryFile.DirectoryName "results.jsonl"
        if ($Summary.question_count -ne 60 -or
            $Summary.operations.success_count -ne 60 -or
            $Summary.operations.failed_count -ne 0 -or
            $Summary.split -eq "test" -or
            $Summary.results_sha256 -ne (Get-Sha256 $ResultPath) -or
            $null -eq $Summary.operations.latency_ms.reranking) {
            throw "S6 saved summary contract mismatch: $($SummaryFile.FullName)"
        }
    }
    $Selection = Get-Content -LiteralPath `
        (Join-Path $ReportsRoot "validation_selected\selection.json") -Raw |
        ConvertFrom-Json
    if ($Selection.selection_split -ne "validation" -or
        $Selection.selected.parameters.rerank_depth -ne 50 -or
        ($Selection.candidate_depths -join ',') -ne "10,20,50") {
        throw "S6 selected Validation depth mismatch"
    }
    $Smoke = Get-Content -LiteralPath `
        (Join-Path $ReportsRoot "real_model_smoke.json") -Raw | ConvertFrom-Json
    if ($Smoke.status -ne "pass" -or $Smoke.process_runs.Count -ne 2 -or
        (($Smoke.process_runs[0] | ConvertTo-Json -Compress) -ne
         ($Smoke.process_runs[1] | ConvertTo-Json -Compress))) {
        throw "S6 committed real-model smoke mismatch"
    }
}

try {
    Set-Location -LiteralPath $ProjectRoot
    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status --short"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before S6 verification:`n$($InitialStatus -join [Environment]::NewLine)"
    }

    Write-Output "[S6] Run S5 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_s5.ps1")
    Assert-LastExitCode "S5 gate"

    Write-Output "[S6] Verify frozen reranker assets offline"
    & uv run python scripts\freeze_s6_reranker.py `
        --model-root $RerankerModelRoot `
        --manifest (Join-Path $ProjectRoot "data\manifests\s6-reranker-model.json")
    Assert-LastExitCode "S6 reranker asset verification"

    Write-Output "[S6] Run deterministic real-model smoke twice"
    $FirstSmoke = (& uv run python scripts\smoke_s6_reranker.py `
        --model-root $RerankerModelRoot | Out-String).Trim()
    Assert-LastExitCode "first S6 real-model smoke"
    $SecondSmoke = (& uv run python scripts\smoke_s6_reranker.py `
        --model-root $RerankerModelRoot | Out-String).Trim()
    Assert-LastExitCode "second S6 real-model smoke"
    if ($FirstSmoke -ne $SecondSmoke) {
        throw "S6 real-model smoke differs across processes"
    }
    $ActualSmoke = $FirstSmoke | ConvertFrom-Json
    $CommittedSmoke = Get-Content -LiteralPath `
        (Join-Path $ProjectRoot "reports\s6_reranking\real_model_smoke.json") -Raw |
        ConvertFrom-Json
    if (($ActualSmoke.scores | ConvertTo-Json -Compress) -ne
        ($CommittedSmoke.process_runs[0] | ConvertTo-Json -Compress)) {
        throw "S6 real-model smoke differs from committed evidence"
    }

    Write-Output "[S6] Validate saved M2 evidence"
    Assert-SavedRunContracts (Join-Path $ProjectRoot "reports\s6_reranking")

    $Manifest = Get-Content -LiteralPath `
        (Join-Path $ProjectRoot "data\eval\s4-golden-v1\manifest.json") -Raw |
        ConvertFrom-Json
    if ($Manifest.test_execution_status -ne "locked_until_s12") {
        throw "S6 requires locked_until_s12 Test status"
    }
    $TempRoot = Join-Path $TempBase ("docguard-s6-verify-" + [Guid]::NewGuid().ToString("N"))
    $LockedOutput = Join-Path $TempRoot "forbidden-s6-test"
    New-Item -ItemType Directory -Path $TempRoot | Out-Null
    $ErrorActionPreference = "Continue"
    try {
        $LockedText = (& uv run python -m docguard.evaluation.cli run `
            --dataset (Join-Path $ProjectRoot "data\eval\s4-golden-v1\test.jsonl") `
            --output $LockedOutput --run-id forbidden-s6-test --method m2 2>&1 |
            Out-String)
        $LockedExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = "Stop"
    }
    if ($LockedExitCode -eq 0 -or $LockedText -notmatch "S12" -or
        (Test-Path -LiteralPath $LockedOutput)) {
        throw "S6 Test split unexpectedly executed"
    }

    Write-Output "[S6] Run S6 unit tests"
    $S6UnitTests = @(
        Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\unit") `
            -File -Filter "test_s6_*.py" |
            Sort-Object FullName |
            ForEach-Object { $_.FullName }
    )
    if ($S6UnitTests.Count -eq 0) {
        throw "No S6 unit tests were discovered"
    }
    & uv run pytest @S6UnitTests -q
    Assert-LastExitCode "S6 tests"

    Write-Output "[S6] Run full regression suite"
    & uv run pytest -q
    Assert-LastExitCode "full tests"
    & uv run ruff check src tests scripts
    Assert-LastExitCode "Ruff"
    & git diff --check
    Assert-LastExitCode "git diff --check"

    Write-Output "[S6] Verify clean worktree"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "git status --short"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S6 verification changed the worktree"
    }
    Write-Output "S6_VERIFY_PASS"
}
catch {
    Write-Error "S6_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
    if ($null -ne $TempRoot -and (Test-Path -LiteralPath $TempRoot)) {
        $SafeTempRoot = [IO.Path]::GetFullPath($TempRoot)
        $ExpectedPrefix = $TempBase + [IO.Path]::DirectorySeparatorChar
        $Leaf = [IO.Path]::GetFileName($SafeTempRoot)
        if (-not $SafeTempRoot.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not $Leaf.StartsWith("docguard-s6-verify-", [StringComparison]::Ordinal)) {
            throw "Refusing to remove unexpected temporary path: $SafeTempRoot"
        }
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
        if (Test-Path -LiteralPath $TempRoot) {
            throw "Failed to remove S6 verification directory: $TempRoot"
        }
    }
}
