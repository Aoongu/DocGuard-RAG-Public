[CmdletBinding()]
param([string]$RerankerModelRoot = "")

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) { $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache" }
$PreviousLocation = Get-Location

function Assert-LastExitCode { param([Parameter(Mandatory = $true)][string]$Step) if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" } }

try {
    Set-Location -LiteralPath $ProjectRoot
    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status"
    if ($InitialStatus.Count -ne 0) { throw "Worktree must be clean before S10 verification" }
    Write-Output "[S10] Run S9 gate"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "scripts\verify_s9.ps1") -RerankerModelRoot $RerankerModelRoot
    Assert-LastExitCode "S9 gate"
    Write-Output "[S10] Run frozen frontend gate"
    Set-Location -LiteralPath (Join-Path $ProjectRoot "frontend")
    & npm ci
    Assert-LastExitCode "npm ci"
    & npm test -- --run
    Assert-LastExitCode "frontend tests"
    & npm run typecheck
    Assert-LastExitCode "frontend typecheck"
    & npm run build
    Assert-LastExitCode "frontend build"
    Set-Location -LiteralPath $ProjectRoot
    $Manifest = Get-Content -LiteralPath (Join-Path $ProjectRoot "data\eval\s4-golden-v1\manifest.json") -Raw | ConvertFrom-Json
    if ($Manifest.test_execution_status -ne "locked_until_s12") { throw "S10 must preserve the Test lock" }
    $CredentialPatterns = @("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "sk-[A-Za-z0-9_-]{20,}")
    $Forbidden = @(Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "frontend\dist") -File -Recurse | Select-String -Pattern $CredentialPatterns)
    if ($Forbidden.Count -ne 0) { throw "frontend bundle contains a forbidden credential marker" }
    & git diff --check
    Assert-LastExitCode "git diff --check"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "final git status"
    if (Compare-Object $InitialStatus $FinalStatus) { throw "S10 verification changed the tracked worktree" }
    Write-Output "S10_VERIFY_PASS"
}
catch { Write-Error "S10_VERIFY_FAIL: $($_.Exception.Message)"; exit 1 }
finally { Set-Location -LiteralPath $PreviousLocation }
