[CmdletBinding()]
param([switch]$NoBuild)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "project_env.ps1")
Import-DocGuardEnvironment -Path (Join-Path $ProjectRoot ".env")
foreach ($Name in @("DEEPSEEK_API_KEY", "SERPAPI_API_KEY", "BRAVE_SEARCH_API_KEY")) {
    if (-not [Environment]::GetEnvironmentVariable($Name, "Process")) {
        [Environment]::SetEnvironmentVariable($Name, [Environment]::GetEnvironmentVariable($Name, "User"), "Process")
    }
}
Push-Location $ProjectRoot
try {
    & docker compose config --quiet
    if ($LASTEXITCODE -ne 0) { throw "Invalid Compose configuration" }
    if (-not $NoBuild) {
        & docker compose build api frontend
        if ($LASTEXITCODE -ne 0) { throw "Image build failed; inspect Docker registry connectivity" }
    }
    & docker compose up -d
    if ($LASTEXITCODE -ne 0) { throw "Container startup failed; run docker compose logs bootstrap api" }
    & docker compose ps
    Write-Host "Open http://127.0.0.1:8080 after services become healthy."
} finally { Pop-Location }
