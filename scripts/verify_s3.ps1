[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PreviousLocation = Get-Location
$TempRoot = $null
$TempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/')

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Get-JsonLines {
    param([Parameter(Mandatory = $true)][string]$Path)

    return @(Get-Content -LiteralPath $Path | ForEach-Object { $_ | ConvertFrom-Json })
}

function Assert-SmokeResults {
    param(
        [Parameter(Mandatory = $true)][object[]]$Expected,
        [Parameter(Mandatory = $true)][object[]]$Actual,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if ($Expected.Count -ne 10 -or $Actual.Count -ne 10) {
        throw "$Label must contain exactly 10 query rows"
    }
    for ($QueryIndex = 0; $QueryIndex -lt 10; $QueryIndex++) {
        $ExpectedRow = $Expected[$QueryIndex]
        $ActualRow = $Actual[$QueryIndex]
        if ($ExpectedRow.query_id -ne $ActualRow.query_id -or
            $ExpectedRow.version -ne $ActualRow.version -or
            $ActualRow.top_k -ne 5 -or
            $ActualRow.hits.Count -ne 5) {
            throw "$Label query contract mismatch at row $QueryIndex"
        }
        for ($Rank = 0; $Rank -lt 5; $Rank++) {
            $ExpectedHit = $ExpectedRow.hits[$Rank]
            $ActualHit = $ActualRow.hits[$Rank]
            $Score = [double]$ActualHit.score
            if ([double]::IsNaN($Score) -or [double]::IsInfinity($Score)) {
                throw "$Label contains a non-finite score"
            }
            if ($null -ne $ActualRow.version -and
                $ActualHit.postgresql_version -ne $ActualRow.version) {
                throw "$Label contains postgresql_version leakage"
            }
            if ($ExpectedHit.chunk_id -ne $ActualHit.chunk_id) {
                throw "$Label Top-K chunk_id mismatch at row $QueryIndex rank $Rank"
            }
            if ([Math]::Abs([double]$ExpectedHit.score - $Score) -gt 0.000001) {
                throw "$Label score mismatch at row $QueryIndex rank $Rank"
            }
        }
    }
}

function Compare-S3Artifacts {
    param(
        [Parameter(Mandatory = $true)][string]$First,
        [Parameter(Mandatory = $true)][string]$Second,
        [Parameter(Mandatory = $true)][string]$ReceiptPath,
        [Parameter(Mandatory = $true)][string]$FirstSmoke,
        [Parameter(Mandatory = $true)][string]$SecondSmoke,
        [Parameter(Mandatory = $true)][string]$OfficialSmoke
    )

    $ReceiptBytes = [IO.File]::ReadAllBytes($ReceiptPath)
    foreach ($Output in @($First, $Second)) {
        $InfoPath = Join-Path $Output "build-info.json"
        $InfoBytes = [IO.File]::ReadAllBytes($InfoPath)
        if ([Convert]::ToBase64String($InfoBytes) -ne [Convert]::ToBase64String($ReceiptBytes)) {
            throw "S3 build-info.json does not match the committed receipt"
        }
        $Info = Get-Content -LiteralPath $InfoPath -Raw | ConvertFrom-Json
        if ($Info.point_count -ne 7413 -or
            $Info.version_counts.'17.11' -ne 3670 -or
            $Info.version_counts.'18.6' -ne 3743 -or
            ($Info.version_counts.'17.11' + $Info.version_counts.'18.6') -ne $Info.point_count) {
            throw "S3 point_count or version_counts mismatch"
        }
    }

    $OfficialRows = Get-JsonLines $OfficialSmoke
    $FirstRows = Get-JsonLines $FirstSmoke
    $SecondRows = Get-JsonLines $SecondSmoke
    Assert-SmokeResults $OfficialRows $FirstRows "first S3 rebuild"
    Assert-SmokeResults $FirstRows $SecondRows "second S3 rebuild"
}

try {
    Set-Location -LiteralPath $ProjectRoot

    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status --short"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before S3 verification:`n$($InitialStatus -join [Environment]::NewLine)"
    }

    Write-Output "[S3] Run S2 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_s2.ps1")
    Assert-LastExitCode "S2 gate"

    Write-Output "[S3] Verify frozen model assets offline"
    & uv run python scripts/freeze_s3_model.py
    Assert-LastExitCode "offline S3 model verification"

    $TempRoot = Join-Path $TempBase ("docguard-s3-verify-" + [Guid]::NewGuid().ToString("N"))
    $TempRaw = Join-Path $TempRoot "raw"
    $CorpusOutput = Join-Path $TempRoot "s2-chunk-v1"
    $FirstOutput = Join-Path $TempRoot "first\s3-dense-v1"
    $SecondOutput = Join-Path $TempRoot "second\s3-dense-v1"
    $FirstSmoke = Join-Path $TempRoot "first-smoke.jsonl"
    $SecondSmoke = Join-Path $TempRoot "second-smoke.jsonl"
    New-Item -ItemType Directory -Path $TempRoot | Out-Null

    Write-Output "[S3] Restore S1 snapshot offline"
    & uv run python -m docguard.ingestion.snapshot restore --output-root $TempRaw
    Assert-LastExitCode "temporary S1 restore"

    Write-Output "[S3] Run temporary S2 build"
    & uv run python -m docguard.processing.pipeline build `
        --raw-root $TempRaw --output-root $CorpusOutput
    Assert-LastExitCode "temporary S2 build"

    $Chunks = Join-Path $CorpusOutput "chunks.jsonl"
    Write-Output "[S3] Run first S3 build"
    & uv run python -m docguard.retrieval.pipeline build `
        --chunks $Chunks --output-root $FirstOutput
    Assert-LastExitCode "first S3 build"

    Write-Output "[S3] Run first S3 verification"
    & uv run python -m docguard.retrieval.pipeline verify `
        --chunks $Chunks --output-root $FirstOutput
    Assert-LastExitCode "first S3 verification"

    & uv run python -m docguard.retrieval.service run-smoke `
        --index-root $FirstOutput --output $FirstSmoke
    Assert-LastExitCode "first S3 smoke run"

    Write-Output "[S3] Run second independent S3 build"
    & uv run python -m docguard.retrieval.pipeline build `
        --chunks $Chunks --output-root $SecondOutput
    Assert-LastExitCode "second independent S3 build"

    & uv run python -m docguard.retrieval.service run-smoke `
        --index-root $SecondOutput --output $SecondSmoke
    Assert-LastExitCode "second S3 smoke run"

    Write-Output "[S3] Compare counts, version filters, and Top-K"
    & Compare-S3Artifacts `
        $FirstOutput `
        $SecondOutput `
        (Join-Path $ProjectRoot "data\manifests\s3-dense-v1.json") `
        $FirstSmoke `
        $SecondSmoke `
        (Join-Path $ProjectRoot "reports\s3_dense_smoke_results.jsonl")

    Write-Output "[S3] Verify clean worktree"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "git status --short"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S3 verification changed the worktree"
    }

    Write-Output "S3_VERIFY_PASS"
}
catch {
    Write-Error "S3_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
    if ($null -ne $TempRoot -and (Test-Path -LiteralPath $TempRoot)) {
        $SafeTempRoot = [IO.Path]::GetFullPath($TempRoot)
        $ExpectedPrefix = $TempBase + [IO.Path]::DirectorySeparatorChar
        $Leaf = [IO.Path]::GetFileName($SafeTempRoot)
        if (-not $SafeTempRoot.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not $Leaf.StartsWith("docguard-s3-verify-", [StringComparison]::Ordinal)) {
            throw "Refusing to remove unexpected temporary path: $SafeTempRoot"
        }
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
        if (Test-Path -LiteralPath $TempRoot) {
            throw "Failed to remove S3 verification directory: $TempRoot"
        }
    }
}
