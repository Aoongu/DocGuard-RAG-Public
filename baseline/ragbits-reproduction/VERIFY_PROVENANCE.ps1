[CmdletBinding()]
param(
    [switch]$UnsafeUtf16RegressionFixture
)

$ErrorActionPreference = "Stop"
$PackageRoot = $PSScriptRoot
$BundlePath = Join-Path $PackageRoot "provenance\ragbits-history.bundle"
$BundleChecksumPath = Join-Path $PackageRoot "provenance\BUNDLE_SHA256.txt"
$MetadataPath = Join-Path $PackageRoot "UPSTREAM_COMMIT.txt"
$UpstreamRoot = Join-Path $PackageRoot "upstream_source"
$ExpectedUpstreamCommit = "32074a71324239ba677234a67737ed65fb678eeb"
$OriginalHistoricalReproductionCommit = "fc411730eb6bb95eaae02fcc7be5c5be34cceac0"
$ExpectedReproductionTree = "99ec16c9efd9fb85e950a1a9b0f71b004eb736e5"
$ExpectedReproductionCommit = "d8ba5d3f00594e0900b7c5c2217935ba0efa2b06"
$ExpectedMainRef = "refs/heads/main"
$ExpectedReproductionRef = "refs/heads/repro/ragbits-sanitized-snapshot-20261011"
$OuterUpstreamPrefix = "baseline/ragbits-reproduction/upstream_source/"
$SafeTempRoot = $null

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $Output = @(& git @Arguments 2>&1)
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed:`n$($Output -join [Environment]::NewLine)"
    }
    return $Output
}

function Read-KeyValueMetadata {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing provenance metadata: $Path"
    }
    $Values = @{}
    foreach ($Line in Get-Content -LiteralPath $Path -Encoding utf8) {
        if ($Line -notmatch "^([^=]+)=(.*)$" -or $Values.ContainsKey($Matches[1])) {
            throw "Invalid provenance metadata line in $Path"
        }
        $Values[$Matches[1]] = $Matches[2]
    }
    return $Values
}

function Read-TreeEntries {
    param(
        [Parameter(Mandatory = $true)][object[]]$Lines,
        [string]$PathPrefix = ""
    )

    $Entries = @{}
    foreach ($LineObject in $Lines) {
        $Line = [string]$LineObject
        if ($Line -notmatch "^([0-9]{6}) blob ([0-9a-f]{40})`t(.+)$") {
            throw "Invalid git tree entry: $Line"
        }
        $Path = $Matches[3]
        if ($PathPrefix) {
            if (-not $Path.StartsWith($PathPrefix, [StringComparison]::Ordinal)) {
                throw "Tree path is outside expected prefix '$PathPrefix': $Path"
            }
            $Path = $Path.Substring($PathPrefix.Length)
        }
        if ($Entries.ContainsKey($Path)) {
            throw "Duplicate git tree path: $Path"
        }
        $Entries[$Path] = @{
            mode = $Matches[1]
            oid = $Matches[2]
        }
    }
    return $Entries
}

function Read-IndexEntries {
    param(
        [Parameter(Mandatory = $true)][object[]]$Lines,
        [Parameter(Mandatory = $true)][string]$PathPrefix
    )

    $Entries = @{}
    foreach ($LineObject in $Lines) {
        $Line = [string]$LineObject
        if ($Line -notmatch "^([0-9]{6}) ([0-9a-f]{40}) 0`t(.+)$") {
            throw "Invalid git index entry: $Line"
        }
        $Path = $Matches[3]
        if (-not $Path.StartsWith($PathPrefix, [StringComparison]::Ordinal)) {
            throw "Index path is outside expected prefix '$PathPrefix': $Path"
        }
        $Path = $Path.Substring($PathPrefix.Length)
        if ($Entries.ContainsKey($Path)) {
            throw "Duplicate git index path: $Path"
        }
        $Entries[$Path] = @{
            mode = $Matches[1]
            oid = $Matches[2]
        }
    }
    return $Entries
}

function Get-GitBlobBytes {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$ObjectId
    )

    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = (Get-Command git).Source
    $StartInfo.WorkingDirectory = $Repository
    $StartInfo.Arguments = "cat-file blob $ObjectId"
    $StartInfo.UseShellExecute = $false
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    $StartInfo.CreateNoWindow = $true
    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $StartInfo
    $null = $Process.Start()
    $Memory = New-Object System.IO.MemoryStream
    $Process.StandardOutput.BaseStream.CopyTo($Memory)
    $ErrorText = $Process.StandardError.ReadToEnd()
    $Process.WaitForExit()
    if ($Process.ExitCode -ne 0) {
        throw "git cat-file blob $ObjectId failed: $ErrorText"
    }
    return ,$Memory.ToArray()
}

function Get-ForbiddenPatterns {
    $ForbiddenPatterns = @(
        @{ label = "Windows user path"; pattern = "(?i)C:\\Users\\" },
        @{ label = "Unix home path"; pattern = "/home/" },
        @{ label = "OpenAI-style API key"; pattern = "sk-[A-Za-z0-9_-]{20,}" },
        @{ label = "Google API key"; pattern = "AIza[0-9A-Za-z_-]{30,}" },
        @{ label = "AWS access key"; pattern = "AKIA[0-9A-Z]{16}" },
        @{ label = "GitHub token"; pattern = "ghp_[A-Za-z0-9]{30,}" },
        @{ label = "private key"; pattern = "-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----" }
    )
    return $ForbiddenPatterns
}

function Assert-SanitizedText {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text,
        [Parameter(Mandatory = $true)][string]$ObjectId,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ScanMode
    )

    foreach ($Forbidden in Get-ForbiddenPatterns) {
        if ($Text -match $Forbidden.pattern) {
            throw "Sensitive pattern '$($Forbidden.label)' in provenance blob $ObjectId ($Path), scan=$ScanMode"
        }
    }
}

function Get-ReliablyDecodedText {
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][byte[]]$Bytes)

    try {
        if ($Bytes.Length -ge 3 -and $Bytes[0] -eq 0xef -and $Bytes[1] -eq 0xbb -and $Bytes[2] -eq 0xbf) {
            $Encoding = New-Object System.Text.UTF8Encoding($false, $true)
            return [PSCustomObject]@{ decoded = $true; label = "UTF-8 BOM"; text = $Encoding.GetString($Bytes, 3, $Bytes.Length - 3) }
        }
        if ($Bytes.Length -ge 2 -and $Bytes[0] -eq 0xff -and $Bytes[1] -eq 0xfe) {
            $Encoding = New-Object System.Text.UnicodeEncoding($false, $true, $true)
            return [PSCustomObject]@{ decoded = $true; label = "UTF-16LE BOM"; text = $Encoding.GetString($Bytes, 2, $Bytes.Length - 2) }
        }
        if ($Bytes.Length -ge 2 -and $Bytes[0] -eq 0xfe -and $Bytes[1] -eq 0xff) {
            $Encoding = New-Object System.Text.UnicodeEncoding($true, $true, $true)
            return [PSCustomObject]@{ decoded = $true; label = "UTF-16BE BOM"; text = $Encoding.GetString($Bytes, 2, $Bytes.Length - 2) }
        }
        if ($Bytes -contains 0) {
            return [PSCustomObject]@{ decoded = $false; label = "NUL without BOM"; text = $null }
        }
        $Encoding = New-Object System.Text.UTF8Encoding($false, $true)
        return [PSCustomObject]@{ decoded = $true; label = "UTF-8"; text = $Encoding.GetString($Bytes) }
    }
    catch [Text.DecoderFallbackException] {
        return [PSCustomObject]@{ decoded = $false; label = "unreliable decoding"; text = $null }
    }
}

function Assert-SanitizedBlobBytes {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][byte[]]$Bytes,
        [Parameter(Mandatory = $true)][string]$ObjectId,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $Decoded = Get-ReliablyDecodedText $Bytes
    if ($Decoded.decoded) {
        Assert-SanitizedText $Decoded.text $ObjectId $Path $Decoded.label
        return "decoded"
    }

    $RawViews = @(
        @{ label = "raw ASCII"; text = [Text.Encoding]::GetEncoding(28591).GetString($Bytes) },
        @{ label = "raw UTF-16LE"; text = [Text.Encoding]::Unicode.GetString($Bytes) },
        @{ label = "raw UTF-16BE"; text = [Text.Encoding]::BigEndianUnicode.GetString($Bytes) }
    )
    foreach ($View in $RawViews) {
        Assert-SanitizedText $View.text $ObjectId $Path $View.label
    }
    return "raw_fallback"
}

function Assert-ProvenanceTextBlobsAreSanitized {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$BaseCommit,
        [Parameter(Mandatory = $true)][string]$TargetCommit
    )

    $BlobPaths = @{}
    $RangeCommits = @(Invoke-Git @("-C", $Repository, "rev-list", "--reverse", "$BaseCommit..$TargetCommit"))
    foreach ($CommitObject in $RangeCommits) {
        $Commit = [string]$CommitObject
        $CommitAndParents = [string](@(Invoke-Git @("-C", $Repository, "rev-list", "--parents", "-n", "1", $Commit))[0])
        $Parts = @($CommitAndParents -split " ")
        if ($Parts.Count -lt 2) {
            throw "Provenance commit has no parent: $Commit"
        }
        $Parent = $Parts[1]
        foreach ($LineObject in Invoke-Git @("-C", $Repository, "diff-tree", "-r", "--no-commit-id", "--diff-filter=AM", "--raw", $Parent, $Commit)) {
            $Line = [string]$LineObject
            if ($Line -notmatch "^:[0-9]{6} [0-9]{6} [0-9a-f]{40} ([0-9a-f]{40}) [AM]`t(.+)$") {
                throw "Invalid provenance diff entry: $Line"
            }
            if (-not $BlobPaths.ContainsKey($Matches[1])) {
                $BlobPaths[$Matches[1]] = $Matches[2]
            }
        }
    }

    $ExplicitBlobAllowlist = @{}
    $DecodedTextScanCount = 0
    $RawByteFallbackScanCount = 0
    $SkippedBlobCount = 0
    foreach ($ObjectId in $BlobPaths.Keys) {
        if ($ExplicitBlobAllowlist.ContainsKey($ObjectId)) {
            $SkippedBlobCount++
            continue
        }
        $Bytes = Get-GitBlobBytes $Repository $ObjectId
        $ScanKind = Assert-SanitizedBlobBytes $Bytes $ObjectId $BlobPaths[$ObjectId]
        if ($ScanKind -eq "decoded") {
            $DecodedTextScanCount++
        }
        elseif ($ScanKind -eq "raw_fallback") {
            $RawByteFallbackScanCount++
        }
        else {
            throw "Unknown provenance blob scan result: $ScanKind"
        }
    }
    if ($BlobPaths.Count -ne ($DecodedTextScanCount + $RawByteFallbackScanCount + $SkippedBlobCount)) {
        throw "Provenance blob scan accounting mismatch"
    }
    return [PSCustomObject]@{
        candidate_blob_count = $BlobPaths.Count
        decoded_text_scan_count = $DecodedTextScanCount
        raw_byte_fallback_scan_count = $RawByteFallbackScanCount
        skipped_blob_count = $SkippedBlobCount
    }
}

try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is required for provenance verification"
    }
    if ($UnsafeUtf16RegressionFixture) {
        $FixturePayload = [Text.Encoding]::Unicode.GetBytes("path=C:\Users\Aoongu\secret.txt")
        $FixtureBytes = [byte[]](0xff, 0xfe) + $FixturePayload
        try {
            $null = Assert-SanitizedBlobBytes `
                $FixtureBytes "regression-fixture" "utf16le-sensitive-path.txt"
        }
        catch {
            if ($_.Exception.Message -notmatch "Sensitive pattern 'Windows user path'.*scan=UTF-16LE BOM") {
                throw
            }
            Write-Output "UTF16LE_REGRESSION_REJECTED"
            exit 0
        }
        throw "UTF-16LE regression fixture was not rejected"
    }
    if (-not (Test-Path -LiteralPath $BundlePath -PathType Leaf)) {
        throw "Missing provenance bundle: $BundlePath"
    }
    if (-not (Test-Path -LiteralPath $BundleChecksumPath -PathType Leaf)) {
        throw "Missing bundle checksum: $BundleChecksumPath"
    }

    $Metadata = Read-KeyValueMetadata $MetadataPath
    if ($Metadata["upstream_commit"] -ne $ExpectedUpstreamCommit -or
        $Metadata["original_historical_reproduction_commit"] -ne $OriginalHistoricalReproductionCommit -or
        $Metadata["reproduction_snapshot_tree"] -ne $ExpectedReproductionTree -or
        $Metadata["reproduction_commit"] -ne $ExpectedReproductionCommit) {
        throw "Immutable provenance metadata does not match verifier constants"
    }
    Write-Output "[PASS] immutable sanitized snapshot metadata"

    $ChecksumLine = (Get-Content -LiteralPath $BundleChecksumPath -Encoding utf8).Trim()
    if ($ChecksumLine -notmatch "^([0-9a-f]{64})  ragbits-history\.bundle$") {
        throw "Invalid BUNDLE_SHA256.txt format"
    }
    $ActualBundleHash = (Get-FileHash -LiteralPath $BundlePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualBundleHash -ne $Matches[1]) {
        throw "Provenance bundle SHA256 mismatch"
    }
    Write-Output "[PASS] provenance bundle SHA256"

    $BundleHeads = @{}
    foreach ($LineObject in Invoke-Git @("bundle", "list-heads", $BundlePath)) {
        $Line = [string]$LineObject
        if ($Line -notmatch "^([0-9a-f]{40}) (refs/heads/.+)$") {
            throw "Invalid bundle ref entry: $Line"
        }
        $BundleHeads[$Matches[2]] = $Matches[1]
    }
    if ($BundleHeads.Count -ne 2 -or
        $BundleHeads[$ExpectedMainRef] -ne $ExpectedUpstreamCommit -or
        $BundleHeads[$ExpectedReproductionRef] -ne $ExpectedReproductionCommit) {
        throw "Bundle refs do not match the fixed upstream and reproduction commits"
    }
    Write-Output "[PASS] exact main and reproduction refs"

    $null = Invoke-Git @("-C", $PackageRoot, "bundle", "verify", $BundlePath)

    $TempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $RequestedTempRoot = Join-Path $TempBase ("docguard-rag-provenance-" + [Guid]::NewGuid().ToString("N"))
    $null = New-Item -ItemType Directory -Path $RequestedTempRoot
    $SafeTempRoot = (Resolve-Path -LiteralPath $RequestedTempRoot).Path
    $SafeTempBase = $TempBase.TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar
    if (-not $SafeTempRoot.StartsWith($SafeTempBase, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe temporary path: $SafeTempRoot"
    }

    $CloneRoot = Join-Path $SafeTempRoot "ragbits"
    $null = Invoke-Git @("clone", "--quiet", "--no-checkout", $BundlePath, $CloneRoot)
    foreach ($Commit in @($ExpectedUpstreamCommit, $ExpectedReproductionCommit)) {
        $null = Invoke-Git @("-C", $CloneRoot, "cat-file", "-e", "$Commit^{commit}")
    }
    $null = Invoke-Git @("-C", $CloneRoot, "fsck", "--full", "--no-dangling")
    Write-Output "[PASS] both commits and complete reachable history"

    $SnapshotCommitLine = [string](@(Invoke-Git @(
        "-C", $CloneRoot, "rev-list", "--parents", "-n", "1", $ExpectedReproductionCommit
    ))[0])
    $SnapshotParts = @($SnapshotCommitLine -split " ")
    if ($SnapshotParts.Count -ne 2 -or $SnapshotParts[1] -ne $ExpectedUpstreamCommit) {
        throw "Sanitized snapshot must have the fixed upstream commit as its only parent"
    }
    $SnapshotTree = [string](@(Invoke-Git @(
        "-C", $CloneRoot, "rev-parse", "$ExpectedReproductionCommit^{tree}"
    ))[0])
    if ($SnapshotTree -ne $ExpectedReproductionTree) {
        throw "Sanitized snapshot tree does not match immutable reproduction tree metadata"
    }
    $SnapshotRangeCount = [int]([string](@(Invoke-Git @(
        "-C", $CloneRoot, "rev-list", "--count", "$ExpectedUpstreamCommit..$ExpectedReproductionCommit"
    ))[0]))
    if ($SnapshotRangeCount -ne 1) {
        throw "Sanitized provenance must contain exactly one reproduction commit"
    }

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $null = & git -C $CloneRoot cat-file -e "$OriginalHistoricalReproductionCommit^{commit}" 2>$null
        $HistoricalCommitExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($HistoricalCommitExitCode -eq 0) {
        throw "Original historical reproduction commit must not be present in the sanitized bundle"
    }
    Write-Output "[PASS] single-parent sanitized snapshot; private historical DAG absent"

    $ScanStats = Assert-ProvenanceTextBlobsAreSanitized `
        $CloneRoot $ExpectedUpstreamCommit $ExpectedReproductionCommit
    Write-Output ("[PASS] provenance blob scan: candidates={0}, decoded_text={1}, raw_byte_fallback={2}, skipped={3}" -f `
        $ScanStats.candidate_blob_count, `
        $ScanStats.decoded_text_scan_count, `
        $ScanStats.raw_byte_fallback_scan_count, `
        $ScanStats.skipped_blob_count)

    $BundleTree = Read-TreeEntries (Invoke-Git @("-C", $CloneRoot, "ls-tree", "-r", $ExpectedUpstreamCommit))
    if ($BundleTree.Count -ne 790) {
        throw "Expected 790 upstream bundle paths, got $($BundleTree.Count)"
    }

    $OuterRootLines = @(Invoke-Git @("-C", $PackageRoot, "rev-parse", "--show-toplevel"))
    if ($OuterRootLines.Count -ne 1) {
        throw "Could not resolve the outer repository root"
    }
    $OuterRoot = [string]$OuterRootLines[0]
    $OuterTree = Read-IndexEntries `
        (Invoke-Git @("-C", $OuterRoot, "ls-files", "--stage", "--", $OuterUpstreamPrefix.TrimEnd("/"))) `
        $OuterUpstreamPrefix
    if ($OuterTree.Count -ne 790) {
        throw "Expected 790 indexed upstream paths, got $($OuterTree.Count)"
    }

    $ArchivedFiles = @(Get-ChildItem -LiteralPath $UpstreamRoot -Recurse -File -Force)
    if ($ArchivedFiles.Count -ne 790) {
        throw "Expected 790 archived upstream files, got $($ArchivedFiles.Count)"
    }
    $ArchivedPaths = @{}
    foreach ($File in $ArchivedFiles) {
        $RelativePath = $File.FullName.Substring($UpstreamRoot.Length + 1).Replace("\", "/")
        $ArchivedPaths[$RelativePath] = $File
    }

    foreach ($Path in $BundleTree.Keys) {
        if (-not $ArchivedPaths.ContainsKey($Path)) {
            throw "Archived upstream tree is missing path: $Path"
        }
        if (-not $OuterTree.ContainsKey($Path)) {
            throw "Outer repository index is missing upstream path: $Path"
        }
        $ExpectedEntry = $BundleTree[$Path]
        $OuterEntry = $OuterTree[$Path]
        if ($OuterEntry.mode -ne $ExpectedEntry.mode) {
            throw "Git mode mismatch for $Path`: expected=$($ExpectedEntry.mode), actual=$($OuterEntry.mode)"
        }
        if ($OuterEntry.oid -ne $ExpectedEntry.oid) {
            throw "Indexed Git blob mismatch for $Path"
        }
        $ArchiveBlobLines = @(Invoke-Git @("hash-object", "--no-filters", "--", $ArchivedPaths[$Path].FullName))
        if ($ArchiveBlobLines.Count -ne 1 -or [string]$ArchiveBlobLines[0] -ne $ExpectedEntry.oid) {
            throw "Archived file Git blob mismatch for $Path"
        }
    }
    Write-Output "[PASS] 790 upstream paths, Git blobs, and modes"

    $SnapshotReproductionTree = Read-TreeEntries `
        (Invoke-Git @("-C", $CloneRoot, "-c", "core.quotepath=false", "ls-tree", "-r", $ExpectedReproductionCommit, "--", "reproduction")) `
        "reproduction/"
    if ($SnapshotReproductionTree.Count -ne 18) {
        throw "Expected 18 reproduction paths in sanitized snapshot, got $($SnapshotReproductionTree.Count)"
    }

    $null = Invoke-Git @("-C", $CloneRoot, "checkout", "--quiet", "--detach", $ExpectedReproductionCommit)
    $CheckedOutHead = @(Invoke-Git @("-C", $CloneRoot, "rev-parse", "HEAD"))
    if ($CheckedOutHead.Count -ne 1 -or [string]$CheckedOutHead[0] -ne $ExpectedReproductionCommit) {
        throw "Reproduction commit checkout failed"
    }
    $CheckedOutReproductionFiles = @(Get-ChildItem -LiteralPath (Join-Path $CloneRoot "reproduction") -Recurse -File -Force)
    if ($CheckedOutReproductionFiles.Count -ne 18) {
        throw "Expected 18 checked-out reproduction files, got $($CheckedOutReproductionFiles.Count)"
    }
    foreach ($File in $CheckedOutReproductionFiles) {
        $RelativePath = $File.FullName.Substring((Join-Path $CloneRoot "reproduction").Length + 1).Replace("\", "/")
        if (-not $SnapshotReproductionTree.ContainsKey($RelativePath)) {
            throw "Unexpected checked-out reproduction path: $RelativePath"
        }
    }
    Write-Output "[PASS] sanitized reproduction snapshot checkout: 18 files"
    Write-Output "PROVENANCE_VERIFY_PASS"
    exit 0
}
catch {
    Write-Error $_ -ErrorAction Continue
    Write-Output "PROVENANCE_VERIFY_FAIL"
    exit 1
}
finally {
    if ($SafeTempRoot -and (Test-Path -LiteralPath $SafeTempRoot)) {
        Remove-Item -LiteralPath $SafeTempRoot -Recurse -Force
    }
}
