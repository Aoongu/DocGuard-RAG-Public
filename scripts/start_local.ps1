[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$AppUrl = "http://127.0.0.1:5173"
$ApiHealthUrl = "http://127.0.0.1:8000/api/v1/health"
$PreviousLocation = Get-Location

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Test-HttpEndpoint {
    param([Parameter(Mandatory = $true)][string]$Uri)
    try {
        $Response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
        return $Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500
    }
    catch {
        return $false
    }
}

function ConvertTo-EncodedCommand {
    param([Parameter(Mandatory = $true)][string]$Command)
    return [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))
}

try {
    Set-Location -LiteralPath $ProjectRoot

    foreach ($CommandName in @("uv", "npm", "powershell")) {
        if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
            throw "Required command is unavailable: $CommandName"
        }
    }

    if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
        $env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable(
            "DEEPSEEK_API_KEY",
            [EnvironmentVariableTarget]::User
        )
    }
    if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
        Write-Warning "DEEPSEEK_API_KEY is unavailable; generated answers will fail closed."
    }

    Write-Host "[1/5] Synchronizing the frozen Python environment..." -ForegroundColor Cyan
    & uv sync --frozen --group dev
    Assert-LastExitCode "uv sync"

    Push-Location -LiteralPath $FrontendRoot
    try {
        Write-Host "[2/5] Installing frozen frontend dependencies..." -ForegroundColor Cyan
        & npm ci
        Assert-LastExitCode "npm ci"

        Write-Host "[3/5] Running frontend tests..." -ForegroundColor Cyan
        & npm test -- --run
        Assert-LastExitCode "frontend tests"

        Write-Host "[4/5] Building the frontend..." -ForegroundColor Cyan
        & npm run build
        Assert-LastExitCode "frontend build"
    }
    finally {
        Pop-Location
    }

    if (Test-HttpEndpoint -Uri $AppUrl) {
        Write-Host "Frontend is already available at $AppUrl" -ForegroundColor Yellow
    }
    else {
        $EscapedFrontendRoot = $FrontendRoot.Replace("'", "''")
        $FrontendCommand = "Set-Location -LiteralPath '$EscapedFrontendRoot'; npm run dev -- --host 127.0.0.1"
        $FrontendEncoded = ConvertTo-EncodedCommand -Command $FrontendCommand
        Start-Process -FilePath "powershell.exe" -ArgumentList @(
            "-NoExit",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-EncodedCommand", $FrontendEncoded
        )
        Write-Host "Started the frontend in a new PowerShell window." -ForegroundColor Green
    }

    $ReadinessCommand = @'
$ErrorActionPreference = "SilentlyContinue"
$deadline = [DateTime]::UtcNow.AddMinutes(2)
while ([DateTime]::UtcNow -lt $deadline) {
    $api = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2
    $frontend = Invoke-WebRequest -Uri "http://127.0.0.1:5173" -UseBasicParsing -TimeoutSec 2
    if ($api.StatusCode -eq 200 -and $frontend.StatusCode -eq 200) {
        Start-Process "http://127.0.0.1:5173"
        exit 0
    }
    Start-Sleep -Seconds 1
}
exit 1
'@
    $ReadinessEncoded = ConvertTo-EncodedCommand -Command $ReadinessCommand
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-EncodedCommand", $ReadinessEncoded
    ) -WindowStyle Hidden

    if (Test-HttpEndpoint -Uri $ApiHealthUrl) {
        Write-Host "API is already available at $ApiHealthUrl" -ForegroundColor Yellow
        Write-Host "Open $AppUrl" -ForegroundColor Green
        return
    }

    Write-Host "[5/5] Starting the API in this PowerShell window..." -ForegroundColor Cyan
    Write-Host "The browser will open $AppUrl after both services are ready." -ForegroundColor Green
    Write-Host "Press Ctrl+C here to stop the API; close the other PowerShell window to stop Vite." -ForegroundColor Yellow
    & uv run docguard-api
}
finally {
    Set-Location -LiteralPath $PreviousLocation
}
