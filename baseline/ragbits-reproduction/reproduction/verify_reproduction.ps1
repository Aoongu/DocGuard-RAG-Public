[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$BaselineCommit = "32074a71324239ba677234a67737ed65fb678eeb"
$ResultFiles = @(
    "reproduction/results/document_search.json",
    "reproduction/results/offline_chat_smoke.json"
)

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Output "[RUN] $Label"
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
    Write-Output "[PASS] $Label"
}

function Test-AllowedAddedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return $Path.StartsWith("reproduction/", [System.StringComparison]::Ordinal) -or
        $Path.StartsWith("docs/superpowers/plans/", [System.StringComparison]::Ordinal)
}

try {
    $RepositoryRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $RepositoryRoot) {
        throw "Unable to locate repository root from reproduction directory"
    }
    Set-Location -LiteralPath $RepositoryRoot
    Write-Output "Repository root: <repository-root>"

    Invoke-NativeChecked "fixed upstream commit exists" "git" @("cat-file", "-e", "$BaselineCommit^{commit}")
    Invoke-NativeChecked "HEAD contains fixed upstream baseline" "git" @("merge-base", "--is-ancestor", $BaselineCommit, "HEAD")
    $ReproductionHead = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $ReproductionHead -notmatch "^[0-9a-f]{40}$") {
        throw "Unable to record current reproduction HEAD"
    }
    Write-Output "Current reproduction HEAD: $ReproductionHead"

    $BaselinePaths = @(& git -c core.quotePath=false ls-tree -r --name-only $BaselineCommit)
    if ($LASTEXITCODE -ne 0 -or $BaselinePaths.Count -eq 0) {
        throw "Unable to enumerate fixed upstream tree"
    }
    $CurrentTrackedPaths = @(& git -c core.quotePath=false ls-files)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to enumerate current tracked paths"
    }
    $UntrackedPaths = @(& git -c core.quotePath=false ls-files --others --exclude-standard)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to enumerate current untracked paths"
    }

    $BaselinePathSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    $CurrentTrackedPathSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($Path in $BaselinePaths) {
        [void]$BaselinePathSet.Add($Path)
    }
    foreach ($Path in $CurrentTrackedPaths) {
        [void]$CurrentTrackedPathSet.Add($Path)
    }

    $MissingBaselinePaths = @($BaselinePaths | Where-Object { -not $CurrentTrackedPathSet.Contains($_) })
    if ($MissingBaselinePaths.Count -ne 0) {
        throw "Fixed upstream paths are missing: $($MissingBaselinePaths -join ', ')"
    }
    $AddedTrackedPaths = @($CurrentTrackedPaths | Where-Object { -not $BaselinePathSet.Contains($_) })
    $DisallowedTrackedPaths = @($AddedTrackedPaths | Where-Object { -not (Test-AllowedAddedPath $_) })
    if ($DisallowedTrackedPaths.Count -ne 0) {
        throw "Tracked additions outside the allowlist: $($DisallowedTrackedPaths -join ', ')"
    }
    $DisallowedUntrackedPaths = @($UntrackedPaths | Where-Object { -not (Test-AllowedAddedPath $_) })
    if ($DisallowedUntrackedPaths.Count -ne 0) {
        throw "Untracked additions outside the allowlist: $($DisallowedUntrackedPaths -join ', ')"
    }

    Invoke-NativeChecked "all fixed upstream paths and clean-filtered blobs match baseline" "git" @(
        "diff", "--exit-code", $BaselineCommit, "--", ".",
        ":(exclude,glob)reproduction/**",
        ":(exclude,glob)docs/superpowers/plans/**"
    )
    Write-Output "[PASS] fixed upstream tree: $($BaselinePaths.Count) paths; additions restricted to reproduction/ and docs/superpowers/plans/"

    foreach ($ResultFile in $ResultFiles) {
        if (-not (Test-Path -LiteralPath $ResultFile -PathType Leaf)) {
            throw "Missing committed result: $ResultFile"
        }
    }
    Invoke-NativeChecked "result JSON files match committed versions before run" "git" @("diff", "--exit-code", "--", $ResultFiles[0], $ResultFiles[1])
    $BeforeHashes = @{}
    foreach ($ResultFile in $ResultFiles) {
        $BeforeHashes[$ResultFile] = (Get-FileHash -Algorithm SHA256 -LiteralPath $ResultFile).Hash
    }

    Invoke-NativeChecked "reproduction tests" "uv" @("run", "--no-sync", "pytest", "reproduction/tests", "-q")
    Invoke-NativeChecked "document search reproduction" "uv" @("run", "--no-sync", "python", "reproduction/run_document_search.py")
    Invoke-NativeChecked "offline chat smoke reproduction" "uv" @("run", "--no-sync", "python", "reproduction/smoke_offline_chat.py")

    # Windows PowerShell 5.1 returns the top-level JSON array as an Object[].
    # Wrapping the pipeline in @() would nest that array and make Count equal 1.
    $DocumentResults = Get-Content -Raw -LiteralPath $ResultFiles[0] | ConvertFrom-Json
    $ExpectedQueries = @(
        @{ Query = "Which Python toolkit coordinates ingestion and semantic retrieval?"; Source = "ragbits_overview.md" },
        @{ Query = "What stores embeddings locally without a database server?"; Source = "vector_store.md" },
        @{ Query = "Which metric checks whether the relevant document ranks first?"; Source = "evaluation.md" }
    )
    if ($DocumentResults.Count -ne $ExpectedQueries.Count) {
        throw "Document result count mismatch: expected 3, got $($DocumentResults.Count)"
    }
    for ($Index = 0; $Index -lt $ExpectedQueries.Count; $Index++) {
        $Item = $DocumentResults[$Index]
        foreach ($Field in @("query", "source", "score", "text")) {
            if ($Field -notin $Item.PSObject.Properties.Name) {
                throw "Document result $Index is missing field '$Field'"
            }
        }
        if ($Item.query -ne $ExpectedQueries[$Index].Query -or $Item.source -ne $ExpectedQueries[$Index].Source) {
            throw "Document result $Index does not match expected query and Top-1 source"
        }
        if ($Item.score -isnot [double] -and $Item.score -isnot [decimal] -and $Item.score -isnot [int]) {
            throw "Document result $Index score is not numeric"
        }
        if ([string]::IsNullOrWhiteSpace([string]$Item.text)) {
            throw "Document result $Index text is empty"
        }
    }
    Write-Output "[PASS] document JSON fields and expected Top-1 sources"

    $ChatResult = Get-Content -Raw -LiteralPath $ResultFiles[1] | ConvertFrom-Json
    foreach ($Field in @("upstream_example_class", "route_count", "requests")) {
        if ($Field -notin $ChatResult.PSObject.Properties.Name) {
            throw "Offline chat result is missing field '$Field'"
        }
    }
    if ($ChatResult.upstream_example_class -ne "examples.chat.offline_chat:MyChat" -or $ChatResult.route_count -le 0) {
        throw "Offline chat result has unexpected upstream class or route count"
    }
    $ExpectedEndpoints = @("/openapi.json", "/api/config")
    if (@($ChatResult.requests).Count -ne 2) {
        throw "Offline chat result must contain exactly two requests"
    }
    for ($Index = 0; $Index -lt $ExpectedEndpoints.Count; $Index++) {
        $Request = @($ChatResult.requests)[$Index]
        foreach ($Field in @("endpoint", "status_code", "response_summary")) {
            if ($Field -notin $Request.PSObject.Properties.Name) {
                throw "Offline chat request $Index is missing field '$Field'"
            }
        }
        if ($Request.endpoint -ne $ExpectedEndpoints[$Index] -or $Request.status_code -ne 200) {
            throw "Offline chat request $Index does not match expected endpoint and HTTP 200"
        }
    }
    Write-Output "[PASS] offline chat JSON fields and expected HTTP 200 responses"

    foreach ($ResultFile in $ResultFiles) {
        $AfterHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ResultFile).Hash
        if ($AfterHash -ne $BeforeHashes[$ResultFile]) {
            throw "Generated result changed after rerun: $ResultFile"
        }
    }
    Invoke-NativeChecked "generated JSON files remain identical to committed versions" "git" @("diff", "--exit-code", "--", $ResultFiles[0], $ResultFiles[1])

    $AddedPaths = @($AddedTrackedPaths + $UntrackedPaths | Sort-Object -Unique)
    $AddedTextFiles = @()
    foreach ($Path in $AddedPaths) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            continue
        }
        $Bytes = [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $Path).Path)
        if ($Bytes -notcontains 0) {
            $AddedTextFiles += Get-Item -LiteralPath $Path
        }
    }
    $SecretPatterns = @(
        "sk-[A-Za-z0-9_-]{20,}",
        "AIza[0-9A-Za-z_-]{30,}",
        "AKIA[0-9A-Z]{16}",
        "ghp_[A-Za-z0-9]{30,}",
        "-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"
    )
    $SecretMatches = $AddedTextFiles | Select-String -Pattern $SecretPatterns
    if ($SecretMatches) {
        throw "A likely literal API key or private key was found in added text files"
    }
    Write-Output "[PASS] no common literal API key/private key patterns in $($AddedTextFiles.Count) added text files"

    Write-Output "VERIFY_PASS"
    exit 0
}
catch {
    Write-Error $_
    Write-Output "VERIFY_FAIL"
    exit 1
}
