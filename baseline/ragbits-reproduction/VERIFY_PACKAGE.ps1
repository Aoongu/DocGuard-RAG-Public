[CmdletBinding()]
param(
    [switch]$WriteReport
)

$ErrorActionPreference = "Stop"
$PackageRoot = $PSScriptRoot
$PackageManifestPath = Join-Path $PackageRoot "SHA256SUMS.txt"
$UpstreamManifestPath = Join-Path $PackageRoot "UPSTREAM_TREE_SHA256.txt"
$MetadataPath = Join-Path $PackageRoot "UPSTREAM_COMMIT.txt"
$ReportPath = Join-Path $PackageRoot "PACKAGE_VERIFICATION.txt"
$ExcludedFromPackageManifest = @("SHA256SUMS.txt", "PACKAGE_VERIFICATION.txt")
$ExpectedUpstreamCommit = "32074a71324239ba677234a67737ed65fb678eeb"
$ExpectedHistoricalReproductionCommit = "fc411730eb6bb95eaae02fcc7be5c5be34cceac0"
$ExpectedReproductionTree = "99ec16c9efd9fb85e950a1a9b0f71b004eb736e5"
$ExpectedReproductionCommit = "d8ba5d3f00594e0900b7c5c2217935ba0efa2b06"

function Get-RelativePackagePath {
    param([Parameter(Mandatory = $true)][string]$FullName)

    return $FullName.Substring($PackageRoot.Length + 1).Replace("\", "/")
}

function Read-KeyValueFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing metadata file: $Path"
    }
    $Values = @{}
    foreach ($Line in Get-Content -LiteralPath $Path -Encoding utf8) {
        if ($Line -notmatch "^([^=]+)=(.*)$") {
            throw "Invalid key=value line in $Path"
        }
        if ($Values.ContainsKey($Matches[1])) {
            throw "Duplicate metadata key '$($Matches[1])' in $Path"
        }
        $Values[$Matches[1]] = $Matches[2]
    }
    return $Values
}

function Read-HashManifest {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing hash manifest: $Path"
    }
    $Values = @{}
    foreach ($Line in Get-Content -LiteralPath $Path -Encoding utf8) {
        if ($Line -notmatch "^([0-9a-f]{64})  (.+)$") {
            throw "Invalid SHA256 manifest line in $Path"
        }
        $RelativePath = $Matches[2]
        if ([System.IO.Path]::IsPathRooted($RelativePath) -or $RelativePath -match "(^|/)\.\.(/|$)") {
            throw "Unsafe relative path in $Path`: $RelativePath"
        }
        if ($Values.ContainsKey($RelativePath)) {
            throw "Duplicate path '$RelativePath' in $Path"
        }
        $Values[$RelativePath] = $Matches[1]
    }
    return $Values
}

function Assert-ManifestMatches {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Manifest,
        [Parameter(Mandatory = $true)][object[]]$Files,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$RelativePathSelector
    )

    $ActualPaths = @($Files | ForEach-Object { & $RelativePathSelector $_ })
    if ($Manifest.Count -ne $ActualPaths.Count) {
        throw "$Label path count mismatch: manifest=$($Manifest.Count), actual=$($ActualPaths.Count)"
    }
    foreach ($Index in 0..($Files.Count - 1)) {
        $File = $Files[$Index]
        $RelativePath = $ActualPaths[$Index]
        if (-not $Manifest.ContainsKey($RelativePath)) {
            throw "$Label is missing path: $RelativePath"
        }
        $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $File.FullName).Hash.ToLowerInvariant()
        if ($ActualHash -ne $Manifest[$RelativePath]) {
            throw "$Label hash mismatch: $RelativePath"
        }
    }
}

try {
    $Metadata = Read-KeyValueFile $MetadataPath
    if ($Metadata["upstream_url"] -ne "https://github.com/deepsense-ai/ragbits" -or
        $Metadata["upstream_commit"] -ne $ExpectedUpstreamCommit -or
        $Metadata["upstream_license"] -ne "MIT" -or
        $Metadata["original_historical_reproduction_commit"] -ne $ExpectedHistoricalReproductionCommit -or
        $Metadata["reproduction_snapshot_tree"] -ne $ExpectedReproductionTree -or
        $Metadata["reproduction_commit"] -ne $ExpectedReproductionCommit) {
        throw "Fixed commit metadata does not match the expected delivery"
    }
    Write-Output "[PASS] fixed upstream and sanitized reproduction snapshot metadata"

    $ForbiddenDirectories = @(Get-ChildItem -LiteralPath $PackageRoot -Recurse -Force -Directory | Where-Object {
        $_.Name -in @(".git", ".venv", "__pycache__", ".pytest_cache") -or
        (Get-RelativePackagePath $_.FullName) -match "^upstream_source/(reproduction|docs/superpowers)(/|$)"
    })
    if ($ForbiddenDirectories.Count -ne 0) {
        throw "Forbidden directories found: $($ForbiddenDirectories.FullName -join ', ')"
    }
    Write-Output "[PASS] forbidden directories absent"

    $PackageManifest = Read-HashManifest $PackageManifestPath
    $PackageFiles = @(Get-ChildItem -LiteralPath $PackageRoot -Recurse -File | Where-Object {
        (Get-RelativePackagePath $_.FullName) -notin $ExcludedFromPackageManifest
    } | Sort-Object { Get-RelativePackagePath $_.FullName })
    Assert-ManifestMatches $PackageManifest $PackageFiles "package manifest" { param($File) Get-RelativePackagePath $File.FullName }
    Write-Output "[PASS] package SHA256 manifest: $($PackageManifest.Count) files"

    $UpstreamRoot = Join-Path $PackageRoot "upstream_source"
    $UpstreamManifest = Read-HashManifest $UpstreamManifestPath
    $UpstreamFiles = @(Get-ChildItem -LiteralPath $UpstreamRoot -Recurse -File | Sort-Object {
        $_.FullName.Substring($UpstreamRoot.Length + 1).Replace("\", "/")
    })
    Assert-ManifestMatches $UpstreamManifest $UpstreamFiles "upstream tree manifest" {
        param($File) $File.FullName.Substring($UpstreamRoot.Length + 1).Replace("\", "/")
    }
    if ($UpstreamManifest.Count -ne 790) {
        throw "Expected 790 upstream files, got $($UpstreamManifest.Count)"
    }
    Write-Output "[PASS] fixed upstream tree SHA256 manifest: 790 files"

    $DocumentResults = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $PackageRoot "reproduction/results/document_search.json") | ConvertFrom-Json
    $ExpectedSources = @("ragbits_overview.md", "vector_store.md", "evaluation.md")
    if ($DocumentResults.Count -ne 3) {
        throw "Document search JSON must contain three results"
    }
    for ($Index = 0; $Index -lt 3; $Index++) {
        $Item = $DocumentResults[$Index]
        foreach ($Field in @("query", "source", "score", "text")) {
            if ($Field -notin $Item.PSObject.Properties.Name) {
                throw "Document search result $Index is missing '$Field'"
            }
        }
        $ScoreIsNumeric = $Item.score -is [double] -or $Item.score -is [decimal] -or $Item.score -is [int]
        if ($Item.source -ne $ExpectedSources[$Index] -or [string]::IsNullOrWhiteSpace([string]$Item.query) -or
            [string]::IsNullOrWhiteSpace([string]$Item.text) -or -not $ScoreIsNumeric) {
            throw "Document search result $Index failed semantic validation"
        }
    }

    $ChatResult = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $PackageRoot "reproduction/results/offline_chat_smoke.json") | ConvertFrom-Json
    if ($ChatResult.upstream_example_class -ne "examples.chat.offline_chat:MyChat" -or $ChatResult.route_count -le 0 -or
        @($ChatResult.requests).Count -ne 2) {
        throw "Offline chat JSON has invalid top-level fields"
    }
    $ExpectedEndpoints = @("/openapi.json", "/api/config")
    for ($Index = 0; $Index -lt 2; $Index++) {
        $Request = @($ChatResult.requests)[$Index]
        foreach ($Field in @("endpoint", "status_code", "response_summary")) {
            if ($Field -notin $Request.PSObject.Properties.Name) {
                throw "Offline chat request $Index is missing '$Field'"
            }
        }
        if ($Request.endpoint -ne $ExpectedEndpoints[$Index] -or $Request.status_code -ne 200) {
            throw "Offline chat request $Index failed semantic validation"
        }
    }
    Write-Output "[PASS] result JSON schema, Top-1 sources, and HTTP 200 semantics"

    $PackageManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PackageManifestPath).Hash.ToLowerInvariant()
    $UpstreamManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $UpstreamManifestPath).Hash.ToLowerInvariant()
    if ($WriteReport) {
        @(
            "package_verify=PASS",
            "final_reproduction_head=$($Metadata['reproduction_commit'])",
            "upstream_commit=$($Metadata['upstream_commit'])",
            "package_hash_entries=$($PackageManifest.Count)",
            "upstream_tree_hash_entries=$($UpstreamManifest.Count)",
            "sha256sums_sha256=$PackageManifestHash",
            "upstream_tree_sha256_manifest_sha256=$UpstreamManifestHash"
        ) | Set-Content -LiteralPath $ReportPath -Encoding utf8
        Write-Output "[PASS] external package verification report written"
    }

    if (Test-Path -LiteralPath $ReportPath -PathType Leaf) {
        $Report = Read-KeyValueFile $ReportPath
        if ($Report["package_verify"] -ne "PASS" -or
            $Report["final_reproduction_head"] -ne $Metadata["reproduction_commit"] -or
            $Report["upstream_commit"] -ne $Metadata["upstream_commit"] -or
            $Report["package_hash_entries"] -ne [string]$PackageManifest.Count -or
            $Report["upstream_tree_hash_entries"] -ne [string]$UpstreamManifest.Count -or
            $Report["sha256sums_sha256"] -ne $PackageManifestHash -or
            $Report["upstream_tree_sha256_manifest_sha256"] -ne $UpstreamManifestHash) {
            throw "PACKAGE_VERIFICATION.txt does not match the current package"
        }
        Write-Output "[PASS] external package verification report matches current package"
    } elseif (-not $WriteReport) {
        throw "PACKAGE_VERIFICATION.txt is missing"
    }

    $SecretPatterns = @(
        "sk-[A-Za-z0-9_-]{20,}",
        "AIza[0-9A-Za-z_-]{30,}",
        "AKIA[0-9A-Z]{16}",
        "ghp_[A-Za-z0-9]{30,}",
        "-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"
    )
    $TextFiles = @()
    foreach ($File in Get-ChildItem -LiteralPath $PackageRoot -Recurse -File) {
        $Bytes = [System.IO.File]::ReadAllBytes($File.FullName)
        if ($Bytes -notcontains 0) {
            $TextFiles += $File
        }
    }
    $SecretMatches = $TextFiles | Select-String -Pattern $SecretPatterns
    if ($SecretMatches) {
        throw "A likely literal API key or private key was found in package text"
    }
    Write-Output "[PASS] no common literal API key/private key patterns in $($TextFiles.Count) package text files"

    Write-Output "Final reproduction HEAD: $($Metadata['reproduction_commit'])"
    Write-Output "SHA256SUMS entries: $($PackageManifest.Count)"
    Write-Output "UPSTREAM_TREE_SHA256 entries: $($UpstreamManifest.Count)"
    Write-Output "PACKAGE_VERIFY_PASS"
    exit 0
}
catch {
    Write-Error $_
    Write-Output "PACKAGE_VERIFY_FAIL"
    exit 1
}
