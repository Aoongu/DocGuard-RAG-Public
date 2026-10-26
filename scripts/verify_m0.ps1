[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PreviousLocation = Get-Location

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Remove-FrozenInterpreterCache {
    $FrozenRoot = [IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot "baseline\ragbits-reproduction\upstream_source\packages\ragbits-core")
    )
    $ProjectPrefix = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/') + `
        [IO.Path]::DirectorySeparatorChar
    if (-not $FrozenRoot.StartsWith($ProjectPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($FrozenRoot) -ne "ragbits-core") {
        throw "Refusing to inspect unexpected frozen source path: $FrozenRoot"
    }
    $ExpectedPrefix = $FrozenRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $Caches = @(Get-ChildItem -LiteralPath $FrozenRoot -Directory -Recurse -Force |
        Where-Object { $_.Name -eq "__pycache__" })
    foreach ($Cache in $Caches) {
        $CachePath = [IO.Path]::GetFullPath($Cache.FullName)
        if (-not $CachePath.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            $Cache.Name -ne "__pycache__") {
            throw "Refusing to remove unexpected interpreter cache: $CachePath"
        }
        Remove-Item -LiteralPath $CachePath -Recurse -Force
    }
}

function Remove-FrozenRuffCache {
    $FrozenRoot = [IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot "baseline\ragbits-reproduction\upstream_source")
    )
    $ProjectPrefix = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/') + `
        [IO.Path]::DirectorySeparatorChar
    if (-not $FrozenRoot.StartsWith($ProjectPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($FrozenRoot) -ne "upstream_source") {
        throw "Refusing to inspect unexpected frozen source path: $FrozenRoot"
    }
    $ExpectedPrefix = $FrozenRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $Caches = @(Get-ChildItem -LiteralPath $FrozenRoot -Directory -Force |
        Where-Object { $_.Name -eq ".ruff_cache" })
    foreach ($Cache in $Caches) {
        $CachePath = [IO.Path]::GetFullPath($Cache.FullName)
        if (-not $CachePath.StartsWith($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            $Cache.Name -ne ".ruff_cache") {
            throw "Refusing to remove unexpected Ruff cache: $CachePath"
        }
        Remove-Item -LiteralPath $CachePath -Recurse -Force
    }
}

try {
    Set-Location -LiteralPath $ProjectRoot

    Write-Output "[M0] Check frozen lockfile"
    & uv lock --check
    Assert-LastExitCode "uv lock --check"

    Write-Output "[M0] Restore frozen Python environment"
    & uv sync --frozen --group dev
    Assert-LastExitCode "uv sync"

    Write-Output "[M0] Run test suite"
    & uv run pytest -q
    Assert-LastExitCode "pytest"

    Write-Output "[M0] Run Ruff"
    & uv run ruff check src tests
    Assert-LastExitCode "ruff"

    Write-Output "[M0] Remove generated interpreter caches from frozen source"
    Remove-FrozenRuffCache
    Remove-FrozenInterpreterCache

    Write-Output "[M0] Verify frozen reproduction package"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        ".\baseline\ragbits-reproduction\VERIFY_PACKAGE.ps1"
    Assert-LastExitCode "package verification"

    Write-Output "[M0] Verify fixed provenance"
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        ".\baseline\ragbits-reproduction\VERIFY_PROVENANCE.ps1"
    Assert-LastExitCode "provenance verification"

    $InitialCommits = @(& git rev-list --max-parents=0 HEAD)
    Assert-LastExitCode "resolve initial commit"
    if ($InitialCommits.Count -ne 1) {
        throw "Expected one initial commit, got $($InitialCommits.Count)"
    }

    Write-Output "[M0] Verify protected reproduction files"
    & git diff --exit-code $InitialCommits[0] -- `
        "baseline/ragbits-reproduction/reproduction" `
        ":(exclude)baseline/ragbits-reproduction/reproduction/SOURCE_MANIFEST.md"
    Assert-LastExitCode "protected reproduction diff"

    Write-Output "[M0] Verify clean worktree"
    $Status = @(& git status --short)
    Assert-LastExitCode "git status"
    if ($Status.Count -ne 0) {
        throw "Worktree is not clean:`n$($Status -join [Environment]::NewLine)"
    }

    Write-Output "M0_VERIFY_PASS"
}
catch {
    Write-Error "M0_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
}
