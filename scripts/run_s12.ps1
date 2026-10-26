[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('offline', 'online', 'all')]
    [string]$Phase,

    [Parameter(Mandatory = $true)]
    [string]$AssetRoot
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ResolvedAssetRoot = (Resolve-Path -LiteralPath $AssetRoot).Path
$env:UV_CACHE_DIR = Join-Path $ProjectRoot '.uv-cache'

$OfflineMethods = @('b0', 'b1', 'm1', 'm2')
$OnlineMethods = @(
    'b2',
    'm3',
    'm4',
    'm4_no_version_filter',
    'm4_no_citation_validation'
)

if ($Phase -in @('online', 'all')) {
    $UserKey = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY', 'User')
    if ([string]::IsNullOrWhiteSpace($UserKey)) {
        throw 'The Windows user-level DEEPSEEK_API_KEY is unavailable.'
    }
    $env:DEEPSEEK_API_KEY = $UserKey
}

$Methods = switch ($Phase) {
    'offline' { $OfflineMethods }
    'online' { $OnlineMethods }
    'all' { $OfflineMethods + $OnlineMethods }
}

foreach ($Method in $Methods) {
    & uv run docguard-s12 run `
        --method $Method `
        --project-root $ProjectRoot `
        --asset-root $ResolvedAssetRoot
    if ($LASTEXITCODE -ne 0) {
        throw "S12 method failed without automatic retry: $Method"
    }
}
