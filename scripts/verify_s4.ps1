[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
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

function Assert-SameBytes {
    param(
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Actual,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $ExpectedHash = (Get-FileHash -LiteralPath $Expected -Algorithm SHA256).Hash
    $ActualHash = (Get-FileHash -LiteralPath $Actual -Algorithm SHA256).Hash
    if ($ExpectedHash -ne $ActualHash) {
        throw "$Label byte hash mismatch"
    }
}

try {
    Set-Location -LiteralPath $ProjectRoot
    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status --short"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before S4 verification:`n$($InitialStatus -join [Environment]::NewLine)"
    }

    Write-Output "[S4] Run S3 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_s3.ps1")
    Assert-LastExitCode "S3 gate"

    $TempRoot = Join-Path $TempBase ("docguard-s4-verify-" + [Guid]::NewGuid().ToString("N"))
    $TempRaw = Join-Path $TempRoot "raw"
    $CorpusOutput = Join-Path $TempRoot "s2-chunk-v1"
    $FirstRoot = Join-Path $TempRoot "first\s4-golden-v1"
    $SecondRoot = Join-Path $TempRoot "second\s4-golden-v1"
    $FirstSmoke = Join-Path $TempRoot "first\s4-evaluator-smoke-v1.jsonl"
    $SecondSmoke = Join-Path $TempRoot "second\s4-evaluator-smoke-v1.jsonl"
    $LockedOutput = Join-Path $TempRoot "forbidden-test-run"
    New-Item -ItemType Directory -Path $TempRoot | Out-Null

    Write-Output "[S4] Restore S1 snapshot offline"
    & uv run python -m docguard.ingestion.snapshot restore --output-root $TempRaw
    Assert-LastExitCode "temporary S1 restore"

    Write-Output "[S4] Build temporary S2 chunks"
    & uv run python -m docguard.processing.pipeline build `
        --raw-root $TempRaw --output-root $CorpusOutput
    Assert-LastExitCode "temporary S2 build"
    $Chunks = Join-Path $CorpusOutput "chunks.jsonl"
    $HumanAuthored = Join-Path $ProjectRoot "data\eval\s4-golden-v1\human_authored.jsonl"
    $ReviewsRound1 = Join-Path $ProjectRoot "data\eval\s4-golden-v1\reviews_round1.jsonl"
    $ReviewsRound2 = Join-Path $ProjectRoot "data\eval\s4-golden-v1\reviews_test_round2.jsonl"

    Write-Output "[S4] Build frozen dataset twice"
    & uv run python scripts/freeze_s4_dataset.py `
        --chunks $Chunks --output-root $FirstRoot --smoke-output $FirstSmoke `
        --human-authored $HumanAuthored --reviews-round1 $ReviewsRound1 `
        --reviews-test-round2 $ReviewsRound2 | Out-Null
    Assert-LastExitCode "first S4 dataset build"
    & uv run python scripts/freeze_s4_dataset.py `
        --chunks $Chunks --output-root $SecondRoot --smoke-output $SecondSmoke `
        --human-authored $HumanAuthored --reviews-round1 $ReviewsRound1 `
        --reviews-test-round2 $ReviewsRound2 | Out-Null
    Assert-LastExitCode "second S4 dataset build"

    foreach ($Name in @("development.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")) {
        Assert-SameBytes (Join-Path $FirstRoot $Name) (Join-Path $SecondRoot $Name) "two S4 builds $Name"
        Assert-SameBytes (Join-Path $ProjectRoot "data\eval\s4-golden-v1\$Name") `
            (Join-Path $FirstRoot $Name) "committed S4 dataset $Name"
    }
    Assert-SameBytes $FirstSmoke $SecondSmoke "two S4 smoke builds"
    Assert-SameBytes (Join-Path $ProjectRoot "data\eval\s4-evaluator-smoke-v1.jsonl") `
        $FirstSmoke "committed S4 smoke dataset"

    $Manifest = Get-Content -LiteralPath (Join-Path $FirstRoot "manifest.json") -Raw | ConvertFrom-Json
    if ($Manifest.status -ne "frozen" -or
        $Manifest.question_count -ne 180 -or
        $Manifest.human_written_count -ne 36 -or
        $Manifest.required_human_written_count -ne 36 -or
        $Manifest.human_review_complete -ne $true -or
        $Manifest.test_double_review_complete -ne $true -or
        $Manifest.test_execution_status -ne "locked_until_s12") {
        throw "S4 frozen-status or human-review boundary mismatch"
    }
    if ($Manifest.review_inputs.'human_authored.jsonl'.count -ne 36 -or
        $Manifest.review_inputs.'reviews_round1.jsonl'.count -ne 180 -or
        $Manifest.review_inputs.'reviews_test_round2.jsonl'.count -ne 60 -or
        $null -eq $Manifest.review_inputs.'human_authored.jsonl'.sha256 -or
        $null -eq $Manifest.review_inputs.'reviews_round1.jsonl'.sha256 -or
        $null -eq $Manifest.review_inputs.'reviews_test_round2.jsonl'.sha256) {
        throw "S4 review inputs are incomplete"
    }
    foreach ($InputName in @("human_authored.jsonl", "reviews_round1.jsonl", "reviews_test_round2.jsonl")) {
        $RecordedHash = $Manifest.review_inputs.$InputName.sha256.ToLower()
        $CommittedHash = (Get-FileHash -LiteralPath `
            (Join-Path $ProjectRoot "data\eval\s4-golden-v1\$InputName") -Algorithm SHA256).Hash.ToLower()
        if ($RecordedHash -ne $CommittedHash) {
            throw "S4 review input hash mismatch: $InputName"
        }
    }

    Write-Output "[S4] Validate source references"
    & uv run python -m docguard.evaluation.cli validate `
        --dataset (Join-Path $FirstRoot "development.jsonl") `
        --dataset (Join-Path $FirstRoot "validation.jsonl") `
        --dataset (Join-Path $FirstRoot "test.jsonl") `
        --chunks $Chunks | Out-Null
    Assert-LastExitCode "S4 source validation"

    $ErrorActionPreference = "Continue"
    try {
        $LockedText = (& uv run python -m docguard.evaluation.cli run `
            --dataset (Join-Path $FirstRoot "test.jsonl") `
            --output $LockedOutput --run-id forbidden-test 2>&1 | Out-String)
        $LockedExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = "Stop"
    }
    if ($LockedExitCode -eq 0 -or $LockedText -notmatch "S12") {
        throw "Test split unexpectedly executed"
    }
    if (Test-Path -LiteralPath $LockedOutput) {
        throw "Test split unexpectedly executed and created output"
    }

    Write-Output "[S4] Run S4 unit and frozen-data tests"
    $S4UnitTests = @(
        Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\unit") `
            -File -Filter "test_s4_*.py" |
            Sort-Object FullName |
            ForEach-Object { $_.FullName }
    )
    if ($S4UnitTests.Count -eq 0) {
        throw "No S4 unit tests were discovered"
    }
    & uv run pytest @S4UnitTests tests/integration/test_s4_frozen_dataset.py -q
    Assert-LastExitCode "S4 tests"
    & uv run ruff check src tests scripts/freeze_s4_dataset.py
    Assert-LastExitCode "Ruff"

    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "final git status --short"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S4 verification changed the worktree"
    }
    Write-Output "S4_VERIFY_PASS"
}
catch {
    Write-Error "S4_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
    if ($null -ne $TempRoot -and (Test-Path -LiteralPath $TempRoot)) {
        $SafeTempRoot = [IO.Path]::GetFullPath($TempRoot)
        $ExpectedPrefix = $TempBase + [IO.Path]::DirectorySeparatorChar
        $Leaf = [IO.Path]::GetFileName($SafeTempRoot)
        if (-not $SafeTempRoot.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not $Leaf.StartsWith("docguard-s4-verify-", [StringComparison]::Ordinal)) {
            throw "Refusing to remove unexpected temporary path: $SafeTempRoot"
        }
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
        if (Test-Path -LiteralPath $TempRoot) {
            throw "Failed to remove S4 verification directory: $TempRoot"
        }
    }
}
