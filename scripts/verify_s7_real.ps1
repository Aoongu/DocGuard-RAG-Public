[CmdletBinding()]
param(
    [string]$RerankerModelRoot = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
}
$PreviousLocation = Get-Location

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-GenerationRun {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$Split,
        [Parameter(Mandatory = $true)][int]$SuccessCount,
        [Parameter(Mandatory = $true)][int]$FailedCount,
        [Parameter(Mandatory = $true)][string]$ResultsSha256
    )

    $Run = Get-Content -LiteralPath (Join-Path $Directory "run.json") -Raw |
        ConvertFrom-Json
    $Summary = Get-Content -LiteralPath (Join-Path $Directory "summary.json") -Raw |
        ConvertFrom-Json
    $ResultsPath = Join-Path $Directory "results.jsonl"
    if ($Run.split -ne $Split -or $Run.route -ne "b2" -or
        $Run.model -ne "deepseek-flash" -or $Run.protocol -ne "deepseek-responses" -or
        $Run.parameters.dense_enabled -ne $true -or
        $Run.parameters.bm25_enabled -ne $false -or
        $Run.parameters.rerank_enabled -ne $false -or
        $Summary.question_count -ne 60 -or
        $Summary.operations.success_count -ne $SuccessCount -or
        $Summary.operations.failed_count -ne $FailedCount -or
        $Summary.results_sha256 -ne $ResultsSha256 -or
        (Get-Sha256 $ResultsPath) -ne $ResultsSha256 -or
        @(Get-Content -LiteralPath $ResultsPath).Count -ne 60) {
        throw "S7 saved generation contract mismatch: $Directory"
    }
}

try {
    Set-Location -LiteralPath $ProjectRoot
    $InitialStatus = @(& git status --short)
    Assert-LastExitCode "initial git status --short"
    if ($InitialStatus.Count -ne 0) {
        throw "Worktree must be clean before S7 real verification:`n$($InitialStatus -join [Environment]::NewLine)"
    }

    Write-Output "[S7 real] Run offline S7 Mock gate"
    $MockGateOutput = @(& powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $ProjectRoot "scripts\verify_s7.ps1") `
        -RerankerModelRoot $RerankerModelRoot)
    Assert-LastExitCode "S7 Mock gate"
    $MockGateOutput | Write-Output
    if ($MockGateOutput -notcontains "S7_MOCK_GATE_PASS") {
        throw "S7 Mock gate marker is missing"
    }

    Write-Output "[S7 real] Validate immutable Development and Validation evidence"
    Assert-GenerationRun `
        -Directory (Join-Path $ProjectRoot "reports\s7_generation\development_b2") `
        -Split "development" -SuccessCount 49 -FailedCount 11 `
        -ResultsSha256 "54387a65b90c6fa720900e5bb465ca5c5f6406e8fe6795f4956d3c2824300cc5"
    Assert-GenerationRun `
        -Directory (Join-Path $ProjectRoot "reports\s7_generation\validation_b2_reproduction_01") `
        -Split "validation" -SuccessCount 44 -FailedCount 16 `
        -ResultsSha256 "0e150ff4b96a9e2999216b2a83f7f83d3a7ac57ee31b656008a03faf702bd62f"

    Write-Output "[S7 real] Validate complete human_reviews.json"
    $HumanPath = Join-Path $ProjectRoot "data\eval\s7-judge-review-v1\human_reviews.json"
    $Human = Get-Content -LiteralPath $HumanPath -Raw | ConvertFrom-Json
    $HumanEvaluable = @($Human.rows | Where-Object { $_.review_status -eq "evaluable" })
    $HumanNonEvaluable = @(
        $Human.rows | Where-Object { $_.review_status -eq "non_evaluable" }
    )
    if ((Get-Sha256 $HumanPath) -ne
            "2e6e10e3dde719b152a52476d54815fc89719cc83b23656a1ddb94a952c4a526" -or
        $Human.schema_version -ne 2 -or
        $Human.human_validation_status -ne "complete" -or
        $Human.reviewer -ne "project_owner" -or
        $Human.template_sha256 -ne
            "44ddc27e269152919baa1e2bf7af7bb81026c1ca77b4d1f7540e8ab8cecdac4d" -or
        $Human.rows.Count -ne 16 -or $HumanEvaluable.Count -ne 15 -or
        $HumanNonEvaluable.Count -ne 1 -or
        $HumanNonEvaluable[0].question_id -ne "s4-dev-032" -or
        $HumanNonEvaluable[0].failure_reason_confirmed_by -ne "project_owner") {
        throw "S7 human review contract mismatch"
    }

    Write-Output "[S7 real] Validate saved Judge identity and agreement"
    $JudgeRoot = Join-Path $ProjectRoot "reports\s7_generation\judge"
    $JudgeRun = Get-Content -LiteralPath (Join-Path $JudgeRoot "run.json") -Raw |
        ConvertFrom-Json
    $JudgeSummary = Get-Content -LiteralPath `
        (Join-Path $JudgeRoot "summary.json") -Raw | ConvertFrom-Json
    $JudgeResultsPath = Join-Path $JudgeRoot "results.jsonl"
    $JudgeRows = @(Get-Content -LiteralPath $JudgeResultsPath | ConvertFrom-Json)
    if ($JudgeRun.split -ne "development" -or
        $JudgeRun.model -ne "deepseek-flash" -or
        $JudgeRun.same_model_as_generator -ne $true -or
        $JudgeRun.independent_prompt_from_generator -ne $true -or
        $JudgeRun.max_retries -ne 0 -or
        $JudgeRun.human_review_sha256 -ne (Get-Sha256 $HumanPath) -or
        $JudgeSummary.question_count -ne 15 -or
        $JudgeSummary.operations.success_count -ne 15 -or
        $JudgeSummary.operations.failed_count -ne 0 -or
        $JudgeSummary.results_sha256 -ne
            "32a2eeabe131e32ed8536c7d083fdb599a14b450cc619bb16cece7b8a3aa1380" -or
        (Get-Sha256 $JudgeResultsPath) -ne $JudgeSummary.results_sha256 -or
        $JudgeRows.Count -ne 15 -or
        @($JudgeRows | Where-Object { $_.status -ne "success" }).Count -ne 0 -or
        $JudgeSummary.agreement.judge_human_intersection_count -ne 15 -or
        $JudgeSummary.agreement.judge_missing_count -ne 0 -or
        $JudgeSummary.agreement.faithfulness.evaluated_count -ne 15 -or
        $JudgeSummary.agreement.answer_relevance.evaluated_count -ne 15) {
        throw "S7 Judge contract mismatch"
    }

    Write-Output "[S7 real] Verify locked_until_s12 and absence of Test runs"
    $Manifest = Get-Content -LiteralPath `
        (Join-Path $ProjectRoot "data\eval\s4-golden-v1\manifest.json") -Raw |
        ConvertFrom-Json
    if ($Manifest.test_execution_status -ne "locked_until_s12") {
        throw "S7 requires locked_until_s12 Test status"
    }
    $S7RunFiles = @(Get-ChildItem -LiteralPath `
        (Join-Path $ProjectRoot "reports\s7_generation") -Recurse -File `
        -Filter "run.json")
    foreach ($RunFile in $S7RunFiles) {
        $SavedRun = Get-Content -LiteralPath $RunFile.FullName -Raw | ConvertFrom-Json
        if ($SavedRun.split -eq "test") {
            throw "S7 Test run artifact found: $($RunFile.FullName)"
        }
    }

    & git diff --check
    Assert-LastExitCode "git diff --check"
    Write-Output "[S7 real] Verify clean worktree"
    $FinalStatus = @(& git status --short)
    Assert-LastExitCode "git status --short"
    if (Compare-Object $InitialStatus $FinalStatus) {
        throw "S7 real verification changed the worktree"
    }
    Write-Output "S7_VERIFY_PASS"
}
catch {
    Write-Error "S7_VERIFY_FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    Set-Location -LiteralPath $PreviousLocation
}
