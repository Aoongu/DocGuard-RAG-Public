[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
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

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-SqliteLogicalDigest {
    param([Parameter(Mandatory = $true)][string]$Path)

    $Digest = & uv run python -c `
        "import sys; from pathlib import Path; from docguard.processing.storage import logical_sqlite_digest; print(logical_sqlite_digest(Path(sys.argv[1])))" `
        $Path
    Assert-LastExitCode "SQLite logical digest"
    return ([string]$Digest).Trim()
}

function Compare-S2Artifacts {
    param(
        [Parameter(Mandatory = $true)][string]$First,
        [Parameter(Mandatory = $true)][string]$Second,
        [Parameter(Mandatory = $true)][string]$ReceiptPath
    )

    $ReceiptBytes = [IO.File]::ReadAllBytes($ReceiptPath)
    $FirstInfoBytes = [IO.File]::ReadAllBytes((Join-Path $First "build-info.json"))
    $SecondInfoBytes = [IO.File]::ReadAllBytes((Join-Path $Second "build-info.json"))
    if ([Convert]::ToBase64String($FirstInfoBytes) -ne [Convert]::ToBase64String($ReceiptBytes) -or
        [Convert]::ToBase64String($SecondInfoBytes) -ne [Convert]::ToBase64String($ReceiptBytes)) {
        throw "S2 build-info.json does not match the committed receipt"
    }

    $Receipt = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
    $FirstChunks = Get-Sha256 (Join-Path $First "chunks.jsonl")
    $SecondChunks = Get-Sha256 (Join-Path $Second "chunks.jsonl")
    if ($FirstChunks -ne $SecondChunks -or $FirstChunks -ne $Receipt.chunks_sha256) {
        throw "chunks_sha256 mismatch"
    }

    $FirstStats = Get-Sha256 (Join-Path $First "stats.json")
    $SecondStats = Get-Sha256 (Join-Path $Second "stats.json")
    if ($FirstStats -ne $SecondStats -or $FirstStats -ne $Receipt.stats_sha256) {
        throw "stats_sha256 mismatch"
    }

    $FirstSqlite = Get-SqliteLogicalDigest (Join-Path $First "metadata.sqlite3")
    $SecondSqlite = Get-SqliteLogicalDigest (Join-Path $Second "metadata.sqlite3")
    if ($FirstSqlite -ne $SecondSqlite -or $FirstSqlite -ne $Receipt.sqlite_logical_sha256) {
        throw "sqlite_logical_sha256 mismatch"
    }
}

try {
    Set-Location -LiteralPath $ProjectRoot

    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status --short"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before S2 verification:`n$($InitialStatus -join [Environment]::NewLine)"
    }

    Write-Output "[S2] Run M1 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_m1.ps1")
    Assert-LastExitCode "M1 gate"

    $TempRoot = Join-Path $TempBase ("docguard-s2-verify-" + [Guid]::NewGuid().ToString("N"))
    $TempRaw = Join-Path $TempRoot "raw"
    $FirstOutput = Join-Path $TempRoot "first\s2-chunk-v1"
    $SecondOutput = Join-Path $TempRoot "second\s2-chunk-v1"
    New-Item -ItemType Directory -Path $TempRoot | Out-Null

    Write-Output "[S2] Restore S1 snapshot offline"
    & uv run python -m docguard.ingestion.snapshot restore --output-root $TempRaw
    Assert-LastExitCode "S1 temporary offline restore"

    Write-Output "[S2] Build first S2 corpus"
    & uv run python -m docguard.processing.pipeline build `
        --raw-root $TempRaw --output-root $FirstOutput
    Assert-LastExitCode "first S2 build"

    Write-Output "[S2] Verify first S2 corpus"
    & uv run python -m docguard.processing.pipeline verify `
        --raw-root $TempRaw --output-root $FirstOutput
    Assert-LastExitCode "first S2 verification"

    Write-Output "[S2] Run second independent S2 build"
    & uv run python -m docguard.processing.pipeline build `
        --raw-root $TempRaw --output-root $SecondOutput
    Assert-LastExitCode "second independent S2 build"

    Write-Output "[S2] Compare deterministic artifacts"
    & Compare-S2Artifacts $FirstOutput $SecondOutput `
        (Join-Path $ProjectRoot "data\manifests\s2-chunk-v1.json")

    Write-Output "[S2] Verify clean worktree"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "git status --short"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S2 verification changed the worktree"
    }

    Write-Output "S2_VERIFY_PASS"
}
catch {
    Write-Error "S2_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
    if ($null -ne $TempRoot -and (Test-Path -LiteralPath $TempRoot)) {
        $SafeTempRoot = [IO.Path]::GetFullPath($TempRoot)
        $ExpectedPrefix = $TempBase + [IO.Path]::DirectorySeparatorChar
        $Leaf = [IO.Path]::GetFileName($SafeTempRoot)
        if (-not $SafeTempRoot.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not $Leaf.StartsWith("docguard-s2-verify-", [StringComparison]::Ordinal)) {
            throw "Refusing to remove unexpected temporary path: $SafeTempRoot"
        }
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
        if (Test-Path -LiteralPath $TempRoot) {
            throw "Failed to remove S2 verification directory: $TempRoot"
        }
    }
}
