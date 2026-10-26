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

function Compare-BM25Artifacts {
    param(
        [Parameter(Mandatory = $true)][string]$FirstRoot,
        [Parameter(Mandatory = $true)][string]$SecondRoot,
        [Parameter(Mandatory = $true)][string]$CommittedManifest,
        [Parameter(Mandatory = $true)][string]$Chunks
    )

    $FirstManifestPath = Join-Path $FirstRoot "manifest.json"
    $SecondManifestPath = Join-Path $SecondRoot "manifest.json"
    Assert-SameBytes $FirstManifestPath $SecondManifestPath "two S5 BM25 manifests"
    Assert-SameBytes $CommittedManifest $FirstManifestPath "committed S5 BM25 manifest"

    & uv run python -m docguard.retrieval.hybrid_pipeline verify `
        --chunks $Chunks --index-root $FirstRoot | Out-Null
    Assert-LastExitCode "first S5 BM25 verification"
    & uv run python -m docguard.retrieval.hybrid_pipeline verify `
        --chunks $Chunks --index-root $SecondRoot | Out-Null
    Assert-LastExitCode "second S5 BM25 verification"

    $Manifest = Get-Content -LiteralPath $FirstManifestPath -Raw | ConvertFrom-Json
    if ($Manifest.document_count -ne 7413 -or
        $Manifest.version_counts.'17.11' -ne 3670 -or
        $Manifest.version_counts.'18.6' -ne 3743 -or
        $Manifest.vocabulary_size -ne 16076 -or
        $Manifest.posting_count -ne 563720 -or
        $Manifest.logical_sha256 -ne "6873f94139cc06e0d3d5a1238709fbdd7f9f0ad497c59a47f2e6bbb579c449cd") {
        throw "S5 BM25 manifest counts or logical_sha256 mismatch"
    }
}

function Get-FixedQueryRows {
    param(
        [Parameter(Mandatory = $true)][string]$IndexRoot,
        [Parameter(Mandatory = $true)][string]$Query,
        [Parameter(Mandatory = $true)][string]$Version
    )

    $QueryArgument = "--query=$Query"
    $Text = (& uv run python -m docguard.retrieval.hybrid_pipeline search `
        --index-root $IndexRoot $QueryArgument --version $Version --top-k 5 | Out-String)
    Assert-LastExitCode "S5 fixed query $Query/$Version"
    return @($Text | ConvertFrom-Json)
}

function Compare-FixedQueries {
    param(
        [Parameter(Mandatory = $true)][string]$FirstRoot,
        [Parameter(Mandatory = $true)][string]$SecondRoot
    )

    $Cases = @(
        @{ Query = "wal_level"; Version = "17.11" },
        @{ Query = "pg_hba.conf"; Version = "18.6" },
        @{ Query = "--max-connections"; Version = "17.11" }
    )
    foreach ($Case in $Cases) {
        $FirstRows = @(Get-FixedQueryRows $FirstRoot $Case.Query $Case.Version)
        $SecondRows = @(Get-FixedQueryRows $SecondRoot $Case.Query $Case.Version)
        if ($FirstRows.Count -eq 0 -or $SecondRows.Count -eq 0) {
            throw "S5 fixed query returned no rows: $($Case.Query)"
        }
        if (($FirstRows | ConvertTo-Json -Depth 20 -Compress) -ne
            ($SecondRows | ConvertTo-Json -Depth 20 -Compress)) {
            throw "S5 fixed query differs across builds: $($Case.Query)"
        }
        foreach ($Row in $FirstRows) {
            if ($Row.postgresql_version -ne $Case.Version) {
                throw "S5 fixed query leaked postgresql_version: $($Case.Query)"
            }
        }
    }
}

function Assert-SavedRunContracts {
    param([Parameter(Mandatory = $true)][string]$ReportsRoot)

    $RunFiles = @(Get-ChildItem -LiteralPath $ReportsRoot -Recurse -File -Filter "run.json")
    $SummaryFiles = @(Get-ChildItem -LiteralPath $ReportsRoot -Recurse -File -Filter "summary.json")
    if ($RunFiles.Count -ne 13 -or $SummaryFiles.Count -ne 13) {
        throw "S5 expected 13 saved runs"
    }
    foreach ($RunFile in $RunFiles) {
        $Run = Get-Content -LiteralPath $RunFile.FullName -Raw | ConvertFrom-Json
        if ($Run.split -eq "test" -or $Run.method -notin @("b0", "b1", "m1")) {
            throw "S5 saved run method/split contract mismatch: $($RunFile.FullName)"
        }
        if ($Run.method -eq "b0" -and ($Run.parameters.dense_enabled -ne $false -or
            $Run.parameters.bm25_enabled -ne $true)) {
            throw "B0 route switches mismatch"
        }
        if ($Run.method -eq "b1" -and ($Run.parameters.dense_enabled -ne $true -or
            $Run.parameters.bm25_enabled -ne $false)) {
            throw "B1 route switches mismatch"
        }
        if ($Run.method -eq "m1" -and ($Run.parameters.dense_enabled -ne $true -or
            $Run.parameters.bm25_enabled -ne $true)) {
            throw "M1 route switches mismatch"
        }
    }
    foreach ($SummaryFile in $SummaryFiles) {
        $Summary = Get-Content -LiteralPath $SummaryFile.FullName -Raw | ConvertFrom-Json
        if ($Summary.question_count -ne 60 -or
            $Summary.operations.success_count -ne 60 -or
            $Summary.operations.failed_count -ne 0 -or
            $Summary.split -eq "test") {
            throw "S5 saved summary contract mismatch: $($SummaryFile.FullName)"
        }
    }
    $Selection = Get-Content -LiteralPath `
        (Join-Path $ReportsRoot "validation_selected\selection.json") -Raw | ConvertFrom-Json
    if ($Selection.selection_split -ne "validation" -or
        $Selection.selected.parameters.dense_depth -ne 50 -or
        $Selection.selected.parameters.bm25_depth -ne 20 -or
        $Selection.selected.parameters.rrf_k -ne 20) {
        throw "S5 selected Validation parameters mismatch"
    }
}

try {
    Set-Location -LiteralPath $ProjectRoot
    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status --short"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before S5 verification:`n$($InitialStatus -join [Environment]::NewLine)"
    }

    Write-Output "[S5] Run S4 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_s4.ps1")
    Assert-LastExitCode "S4 gate"

    $TempRoot = Join-Path $TempBase ("docguard-s5-verify-" + [Guid]::NewGuid().ToString("N"))
    $TempRaw = Join-Path $TempRoot "raw"
    $CorpusOutput = Join-Path $TempRoot "s2-chunk-v1"
    $FirstOutput = Join-Path $TempRoot "first\s5-bm25-v1"
    $SecondOutput = Join-Path $TempRoot "second\s5-bm25-v1"
    $LockedOutput = Join-Path $TempRoot "forbidden-test-run"
    New-Item -ItemType Directory -Path $TempRoot | Out-Null

    Write-Output "[S5] Restore S1 snapshot offline"
    & uv run python -m docguard.ingestion.snapshot restore --output-root $TempRaw
    Assert-LastExitCode "temporary S1 restore"

    Write-Output "[S5] Build temporary S2 chunks"
    & uv run python -m docguard.processing.pipeline build `
        --raw-root $TempRaw --output-root $CorpusOutput
    Assert-LastExitCode "temporary S2 build"
    $Chunks = Join-Path $CorpusOutput "chunks.jsonl"

    Write-Output "[S5] Run first S5 BM25 build"
    & uv run python -m docguard.retrieval.hybrid_pipeline build `
        --chunks $Chunks --output-root $FirstOutput | Out-Null
    Assert-LastExitCode "first S5 BM25 build"

    Write-Output "[S5] Run second S5 BM25 build"
    & uv run python -m docguard.retrieval.hybrid_pipeline build `
        --chunks $Chunks --output-root $SecondOutput | Out-Null
    Assert-LastExitCode "second S5 BM25 build"

    Write-Output "[S5] Compare deterministic BM25 artifacts"
    & Compare-BM25Artifacts `
        $FirstOutput $SecondOutput `
        (Join-Path $ProjectRoot "data\manifests\s5-bm25-v1.json") $Chunks

    Write-Output "[S5] Compare fixed exact-token queries and version filters"
    & Compare-FixedQueries $FirstOutput $SecondOutput

    Write-Output "[S5] B0/B1/M1 contract checks"
    Assert-SavedRunContracts (Join-Path $ProjectRoot "reports\s5_retrieval")

    $ErrorActionPreference = "Continue"
    try {
        $LockedText = (& uv run python -m docguard.evaluation.cli run `
            --dataset (Join-Path $ProjectRoot "data\eval\s4-golden-v1\test.jsonl") `
            --output $LockedOutput --run-id forbidden-s5-test --method m1 2>&1 | Out-String)
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

    Write-Output "[S5] Run S5 unit tests"
    $S5UnitTests = @(
        Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tests\unit") `
            -File -Filter "test_s5_*.py" |
            Sort-Object FullName |
            ForEach-Object { $_.FullName }
    )
    if ($S5UnitTests.Count -eq 0) {
        throw "No S5 unit tests were discovered"
    }
    & uv run pytest @S5UnitTests -q
    Assert-LastExitCode "S5 tests"
    & uv run ruff check src tests
    Assert-LastExitCode "Ruff"

    Write-Output "[S5] Verify clean worktree"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "git status --short"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S5 verification changed the worktree"
    }
    Write-Output "S5_VERIFY_PASS"
}
catch {
    Write-Error "S5_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
    if ($null -ne $TempRoot -and (Test-Path -LiteralPath $TempRoot)) {
        $SafeTempRoot = [IO.Path]::GetFullPath($TempRoot)
        $ExpectedPrefix = $TempBase + [IO.Path]::DirectorySeparatorChar
        $Leaf = [IO.Path]::GetFileName($SafeTempRoot)
        if (-not $SafeTempRoot.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not $Leaf.StartsWith("docguard-s5-verify-", [StringComparison]::Ordinal)) {
            throw "Refusing to remove unexpected temporary path: $SafeTempRoot"
        }
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
        if (Test-Path -LiteralPath $TempRoot) {
            throw "Failed to remove S5 verification directory: $TempRoot"
        }
    }
}
