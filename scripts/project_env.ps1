function Import-DocGuardEnvironment {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    $AllowedNames = @(
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
        "SERPAPI_API_KEY", "BRAVE_SEARCH_API_KEY", "DOCGUARD_SEARCH_PROVIDER",
        "DOCGUARD_DOCUMENT_DNS", "DOCGUARD_DOH_PROVIDER", "DOCGUARD_DISCOVERY_ROOT",
        "DOCGUARD_PROJECT_ROOT", "DOCGUARD_ASSET_ROOT"
    )
    $LineNumber = 0
    foreach ($Line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $LineNumber += 1
        $Text = $Line.Trim()
        if (-not $Text -or $Text.StartsWith("#")) { continue }
        if ($Text -notmatch '^([A-Z][A-Z0-9_]*)\s*=\s*(.*)$') {
            throw "Invalid .env syntax at line $LineNumber"
        }
        $VariableName = $Matches[1]
        $VariableValue = $Matches[2].Trim()
        if ($AllowedNames -notcontains $VariableName) { continue }
        if ($VariableValue.Length -ge 2) {
            if (($VariableValue.StartsWith('"') -and $VariableValue.EndsWith('"')) -or
                ($VariableValue.StartsWith("'") -and $VariableValue.EndsWith("'"))) {
                $VariableValue = $VariableValue.Substring(1, $VariableValue.Length - 2)
            }
        }
        # Blank fields keep the existing process/user credential fallback.
        # Values are literal strings; no interpolation, dot-sourcing or evaluation.
        if (-not [string]::IsNullOrWhiteSpace($VariableValue)) {
            [Environment]::SetEnvironmentVariable($VariableName, $VariableValue, "Process")
        }
    }
}
