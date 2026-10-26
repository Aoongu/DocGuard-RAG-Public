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

function Assert-FileIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][long]$ExpectedLength,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )

    $Path = Join-Path $ProjectRoot $RelativePath
    $File = Get-Item -LiteralPath $Path -Force
    if ($File.PSIsContainer -or $File.Length -ne $ExpectedLength) {
        throw "$RelativePath has an unexpected type or length"
    }
    $ActualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $ExpectedSha256) {
        throw "$RelativePath SHA-256 mismatch"
    }
}

function Get-HtmlSnapshot {
    param([Parameter(Mandatory = $true)][string]$Root)

    return @(
        Get-ChildItem -LiteralPath $Root -Filter "*.html" -File -Recurse |
            Sort-Object FullName |
            ForEach-Object {
                $RelativePath = $_.FullName.Substring($Root.Length).TrimStart('\', '/')
                "$RelativePath|$($_.Length)|$($_.LastWriteTimeUtc.Ticks)|$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash)"
            }
    )
}

try {
    Set-Location -LiteralPath $ProjectRoot

    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status --short"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before M1 verification:`n$($InitialStatus -join [Environment]::NewLine)"
    }

    Write-Output "[M1] Run M0 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_m0.ps1")
    Assert-LastExitCode "M0 gate"

    Write-Output "[M1] Verify frozen archives and manifests"
    Assert-FileIdentity "data\archives\postgresql-17.11-docs.tar.gz" 3843559 `
        "cd1dbddebbd34ab038f6225a3f66411d1ed6e16b0e7f4b8cfa6f45e99642fa77"
    Assert-FileIdentity "data\archives\postgresql-18.6-docs.tar.gz" 3906887 `
        "0419dec0d3b7ca55a80c0519a1dd88a8d172019ef9841288889f0281ea1f97ed"
    Assert-FileIdentity "data\manifests\sources.json" 6677 `
        "c7a1dd19e3e90c4fc22301147964406ca9becb26481b732a42acbc19da7b2619"
    Assert-FileIdentity "data\manifests\snapshot.jsonl" 447720 `
        "c493a002cf22b32c3be25af6a25d0e35a92d0d97be9491f971557d99a7771da7"
    $ManifestLines = @(Get-Content -LiteralPath "data\manifests\snapshot.jsonl").Count
    if ($ManifestLines -ne 773) {
        throw "snapshot.jsonl expected 773 records, got $ManifestLines"
    }

    $TempRoot = Join-Path $TempBase ("docguard-m1-verify-" + [Guid]::NewGuid().ToString("N"))
    $TempRaw = Join-Path $TempRoot "raw"
    New-Item -ItemType Directory -Path $TempRoot | Out-Null

    Write-Output "[M1] Restore frozen snapshot offline"
    & uv run python -m docguard.ingestion.snapshot restore --output-root $TempRaw
    Assert-LastExitCode "offline snapshot restore"

    $AllHtml = @(Get-ChildItem -LiteralPath $TempRaw -Filter "*.html" -File -Recurse)
    $Version17 = @(Get-ChildItem -LiteralPath (Join-Path $TempRaw "postgresql-17.11") -Filter "*.html" -File).Count
    $Version18 = @(Get-ChildItem -LiteralPath (Join-Path $TempRaw "postgresql-18.6") -Filter "*.html" -File).Count
    if ($AllHtml.Count -ne 773 -or $Version17 -ne 386 -or $Version18 -ne 387) {
        throw "restored snapshot counts are not 773/386/387"
    }

    Write-Output "[M1] Verify restored snapshot"
    & uv run python -m docguard.ingestion.snapshot verify --output-root $TempRaw
    Assert-LastExitCode "restored snapshot verification"
    $BeforeSecondRestore = Get-HtmlSnapshot $TempRaw

    Write-Output "[M1] Verify idempotent offline restore"
    & uv run python -m docguard.ingestion.snapshot restore --output-root $TempRaw
    Assert-LastExitCode "idempotent offline snapshot restore"
    $AfterSecondRestore = Get-HtmlSnapshot $TempRaw
    if (Compare-Object $BeforeSecondRestore $AfterSecondRestore) {
        throw "second restore changed the frozen snapshot"
    }

    & uv run python -m docguard.ingestion.snapshot verify --output-root $TempRaw
    Assert-LastExitCode "second restored snapshot verification"

    Write-Output "[M1] Verify clean worktree"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "git status --short"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "M1 verification changed the worktree"
    }

    Write-Output "M1_VERIFY_PASS"
}
catch {
    Write-Error "M1_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
    if ($null -ne $TempRoot -and (Test-Path -LiteralPath $TempRoot)) {
        $SafeTempRoot = [IO.Path]::GetFullPath($TempRoot)
        $ExpectedPrefix = $TempBase + [IO.Path]::DirectorySeparatorChar
        $Leaf = [IO.Path]::GetFileName($SafeTempRoot)
        if (-not $SafeTempRoot.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not $Leaf.StartsWith("docguard-m1-verify-", [StringComparison]::Ordinal)) {
            throw "Refusing to remove unexpected temporary path: $SafeTempRoot"
        }
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
        if (Test-Path -LiteralPath $TempRoot) {
            throw "Failed to remove M1 verification directory: $TempRoot"
        }
    }
}
