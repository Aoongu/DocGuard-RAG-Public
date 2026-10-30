import subprocess
import tomllib
from pathlib import Path, PurePosixPath

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = PROJECT_ROOT / "baseline" / "ragbits-reproduction"
UPSTREAM_PREFIX = "baseline/ragbits-reproduction/upstream_source/"
UPSTREAM_COMMIT = "32074a71324239ba677234a67737ed65fb678eeb"
REQUIRED_DIRECTORIES = (
    "src/docguard",
    "configs",
    "data",
    "reports",
    "frontend",
    "scripts",
)
REQUIRED_RESPONSIBILITY_DOCS = (
    "README.md",
    "configs/README.md",
    "data/README.md",
    "reports/README.md",
    "frontend/README.md",
)
FORBIDDEN_DEPENDENCIES = {
    "ragbits-agents",
    "ragbits-chat",
    "ragbits-evaluate",
    "ragbits-guardrails",
    "ray",
}


def run_git(*arguments: str, cwd: Path = PROJECT_ROOT) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def parse_tree_entries(output: str, prefix: str = "") -> dict[str, tuple[str, str]]:
    entries = {}
    for line in output.splitlines():
        metadata, path = line.split("\t", maxsplit=1)
        fields = metadata.split()
        mode = fields[0]
        object_id = fields[2] if fields[1] == "blob" else fields[1]
        if prefix:
            assert path.startswith(prefix)
            path = path.removeprefix(prefix)
        entries[path] = (mode, object_id)
    return entries


def test_required_project_directories_and_responsibility_docs_exist() -> None:
    for relative_path in REQUIRED_DIRECTORIES:
        directory = PROJECT_ROOT / relative_path
        assert directory.is_dir(), f"missing project directory: {relative_path}"
    for relative_path in REQUIRED_RESPONSIBILITY_DOCS:
        assert (PROJECT_ROOT / relative_path).is_file(), (
            f"missing responsibility document: {relative_path}"
        )


def test_only_approved_direct_runtime_dependencies_are_declared() -> None:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8").lower()
    for forbidden_dependency in FORBIDDEN_DEPENDENCIES:
        assert forbidden_dependency not in pyproject_text

    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)

    dependencies = set(pyproject["project"]["dependencies"])
    normalized_names = {dependency.split("=", maxsplit=1)[0] for dependency in dependencies}
    assert normalized_names == {
        "fastapi",
        "httpx",
        "lxml",
        "langgraph",
        "numpy",
        "onnxruntime",
        "qdrant-client",
        "ragbits-core",
        "tokenizers",
        "uvicorn",
    }
    assert normalized_names.isdisjoint(FORBIDDEN_DEPENDENCIES)


def test_reproduction_files_match_the_initial_commit_except_manifest() -> None:
    initial_commit = run_git("rev-list", "--max-parents=0", "HEAD").strip()
    changed_paths = run_git(
        "diff",
        "--name-only",
        initial_commit,
        "--",
        "baseline/ragbits-reproduction/reproduction",
        ":(exclude)baseline/ragbits-reproduction/reproduction/SOURCE_MANIFEST.md",
    )
    assert changed_paths == ""


def test_m0_verifier_checks_the_lockfile_before_syncing() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_m0.ps1").read_text(
        encoding="utf-8"
    )
    lock_command = "& uv lock --check"
    lock_exit_check = 'Assert-LastExitCode "uv lock --check"'
    sync_command = "& uv sync --frozen --group dev"

    assert lock_command in verifier
    assert lock_exit_check in verifier
    assert verifier.index(lock_command) < verifier.index(lock_exit_check)
    assert verifier.index(lock_exit_check) < verifier.index(sync_command)


def test_m1_verifier_runs_m0_then_offline_restore_and_clean_check() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_m1.ps1").read_text(
        encoding="utf-8"
    )

    required_steps = (
        "scripts\\verify_m0.ps1",
        "docguard.ingestion.snapshot restore",
        "docguard.ingestion.snapshot verify",
        "git status --short",
        "M1_VERIFY_PASS",
    )
    positions = [
        verifier.rindex(step) if step == "git status --short" else verifier.index(step)
        for step in required_steps
    ]

    assert positions == sorted(positions)
    assert '--download-missing' not in verifier
    assert 'try {' in verifier
    assert 'finally {' in verifier
    assert 'Remove-Item -LiteralPath $TempRoot -Recurse -Force' in verifier


def test_s2_verifier_runs_m1_then_two_offline_builds_and_clean_check() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_s2.ps1").read_text(
        encoding="utf-8"
    )

    required_steps = (
        "scripts\\verify_m1.ps1",
        "docguard.ingestion.snapshot restore",
        "docguard.processing.pipeline build",
        "docguard.processing.pipeline verify",
        "second independent S2 build",
        "& Compare-S2Artifacts",
        "git status --short",
        "S2_VERIFY_PASS",
    )
    positions = [
        verifier.rindex(step) if step == "git status --short" else verifier.index(step)
        for step in required_steps
    ]

    assert positions == sorted(positions)
    assert "--download-missing" not in verifier
    assert '$ErrorActionPreference = "Stop"' in verifier
    assert "Assert-LastExitCode" in verifier
    assert "try {" in verifier
    assert "finally {" in verifier
    assert "chunks_sha256" in verifier
    assert "sqlite_logical_sha256" in verifier
    assert "stats_sha256" in verifier
    assert "Remove-Item -LiteralPath $TempRoot -Recurse -Force" in verifier


def test_s3_verifier_runs_s2_then_two_offline_indexes_and_clean_check() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_s3.ps1").read_text(
        encoding="utf-8"
    )

    required_steps = (
        "git status --short",
        "scripts\\verify_s2.ps1",
        "scripts/freeze_s3_model.py",
        "docguard.processing.pipeline build",
        "first S3 build",
        "first S3 verification",
        "second independent S3 build",
        "& Compare-S3Artifacts",
        "git status --short",
        "S3_VERIFY_PASS",
    )
    positions = [
        verifier.index(step) if index == 0 else verifier.rindex(step)
        if index == 8
        else verifier.index(step)
        for index, step in enumerate(required_steps)
    ]

    assert positions == sorted(positions)
    assert "--download" not in verifier
    assert "snapshot_download" not in verifier
    assert "--download-missing" not in verifier
    assert '$ErrorActionPreference = "Stop"' in verifier
    assert "Assert-LastExitCode" in verifier
    assert "try {" in verifier
    assert "finally {" in verifier
    assert "point_count" in verifier
    assert "version_counts" in verifier
    assert "postgresql_version" in verifier
    assert "[Math]::Abs" in verifier
    assert "Remove-Item -LiteralPath $TempRoot -Recurse -Force" in verifier


def test_readme_describes_the_implemented_m1_boundary() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    data_readme = (PROJECT_ROOT / "data" / "README.md").read_text(
        encoding="utf-8"
    )

    for expected in (
        "PostgreSQL 17.11",
        "PostgreSQL 18.6",
        "773",
        "verify_m1.ps1",
        "--download-missing",
        "S3",
    ):
        assert expected in readme

    for expected in (
        "386",
        "387",
        "data/raw",
        "staging",
        "六个技术主题",
    ):
        assert expected in data_readme


def test_s4_verifier_chains_prior_gate_and_locks_test_split() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_s4.ps1").read_text(
        encoding="utf-8"
    )

    required_steps = (
        "scripts\\verify_s3.ps1",
        "docguard.ingestion.snapshot restore",
        "docguard.processing.pipeline build",
        "scripts/freeze_s4_dataset.py",
        "docguard.evaluation.cli validate",
        "$S4UnitTests = @(",
        "tests/integration/test_s4_frozen_dataset.py",
        "S4_VERIFY_PASS",
    )
    positions = [verifier.index(step) for step in required_steps]
    assert positions == sorted(positions)
    assert '$ErrorActionPreference = "Stop"' in verifier
    assert "Assert-LastExitCode" in verifier
    assert "locked_until_s12" in verifier
    assert "test.jsonl" in verifier
    assert "Test split unexpectedly executed" in verifier
    assert '$ErrorActionPreference = "Continue"' in verifier
    assert "$LockedExitCode = $LASTEXITCODE" in verifier
    assert '$ErrorActionPreference = "Stop"' in verifier
    assert '-Filter "test_s4_*.py"' in verifier
    assert "@S4UnitTests" in verifier
    assert "No S4 unit tests were discovered" in verifier
    for review_input in (
        "human_authored.jsonl",
        "reviews_round1.jsonl",
        "reviews_test_round2.jsonl",
    ):
        assert f"$Manifest.review_inputs.'{review_input}'.count" in verifier
    assert "Remove-Item -LiteralPath $TempRoot -Recurse -Force" in verifier
    assert "--download-missing" not in verifier


def test_s5_verifier_chains_s4_rebuilds_bm25_and_checks_clean_state() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_s5.ps1").read_text(
        encoding="utf-8"
    )

    required_steps = (
        "scripts\\verify_s4.ps1",
        "docguard.ingestion.snapshot restore",
        "docguard.processing.pipeline build",
        "first S5 BM25 build",
        "second S5 BM25 build",
        "Compare-BM25Artifacts",
        "Compare-FixedQueries",
        "B0/B1/M1 contract checks",
        "$S5UnitTests = @(",
        "git status --short",
        "S5_VERIFY_PASS",
    )
    positions = [
        verifier.rindex(step) if index in {5, 6, 9} else verifier.index(step)
        for index, step in enumerate(required_steps)
    ]
    assert positions == sorted(positions)
    assert '$ErrorActionPreference = "Stop"' in verifier
    assert "Assert-LastExitCode" in verifier
    assert "data\\manifests\\s5-bm25-v1.json" in verifier
    assert "logical_sha256" in verifier
    assert "wal_level" in verifier
    assert "pg_hba.conf" in verifier
    assert "--max-connections" in verifier
    assert "postgresql_version" in verifier
    assert "Test split unexpectedly executed" in verifier
    assert '-Filter "test_s5_*.py"' in verifier
    assert "No S5 unit tests were discovered" in verifier
    assert "docguard-s5-verify-" in verifier
    assert "[Guid]::NewGuid()" in verifier
    assert "Remove-Item -LiteralPath $TempRoot -Recurse -Force" in verifier
    assert "--download-missing" not in verifier


def test_s6_verifier_chains_s5_checks_model_smoke_evidence_and_clean_state() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_s6.ps1").read_text(
        encoding="utf-8"
    )

    required_steps = (
        "scripts\\verify_s5.ps1",
        "scripts\\freeze_s6_reranker.py",
        "scripts\\smoke_s6_reranker.py",
        "Assert-SavedRunContracts",
        "forbidden-s6-test",
        "$S6UnitTests = @(",
        "uv run pytest -q",
        "uv run ruff check src tests scripts",
        "git diff --check",
        "git status --short",
        "S6_VERIFY_PASS",
    )
    positions = [
        verifier.rindex(step) if index in {3, 9} else verifier.index(step)
        for index, step in enumerate(required_steps)
    ]

    assert positions == sorted(positions)
    assert '$ErrorActionPreference = "Stop"' in verifier
    assert "Assert-LastExitCode" in verifier
    assert "--download" not in verifier
    assert "locked_until_s12" in verifier
    assert "Remove-Item -LiteralPath $TempRoot -Recurse -Force" in verifier


def test_s7_mock_verifier_chains_s6_and_keeps_real_completion_open() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_s7.ps1").read_text(
        encoding="utf-8"
    )

    required_steps = (
        "scripts\\verify_s6.ps1",
        "docguard.evaluation.generation_cli validate-config",
        "two independent Mock smokes",
        "results_sha256",
        "tests\\integration\\test_s7_mock_generation.py",
        "forbidden-s7-test",
        '$S7UnitTests = @(',
        "uv run pytest -q",
        "uv run ruff check src tests scripts",
        "git diff --check",
        "git status --short",
        "S7_MOCK_GATE_PASS",
    )
    positions = [
        verifier.rindex(step) if step == "git status --short" else verifier.index(step)
        for step in required_steps
    ]
    assert positions == sorted(positions)
    assert '$ErrorActionPreference = "Stop"' in verifier
    assert "Assert-LastExitCode" in verifier
    assert "locked_until_s12" in verifier
    assert "S7_VERIFY_PASS" not in verifier
    assert "OPENAI_API_KEY" not in verifier
    assert "api.openai.com" not in verifier
    assert "Remove-Item -LiteralPath $TempRoot -Recurse -Force" in verifier


def test_s7_real_verifier_freezes_real_runs_human_review_and_judge() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_s7_real.ps1").read_text(
        encoding="utf-8"
    )

    required_steps = (
        "scripts\\verify_s7.ps1",
        "development_b2",
        "validation_b2_reproduction_01",
        "human_reviews.json",
        "reports\\s7_generation\\judge",
        "judge_human_intersection_count",
        "locked_until_s12",
        "git diff --check",
        "git status --short",
        "S7_VERIFY_PASS",
    )
    positions = [
        verifier.rindex(step) if step == "git status --short" else verifier.index(step)
        for step in required_steps
    ]
    assert positions == sorted(positions)
    assert '$ErrorActionPreference = "Stop"' in verifier
    assert "S7_MOCK_GATE_PASS" in verifier
    assert "same_model_as_generator" in verifier
    assert "independent_prompt_from_generator" in verifier
    assert "test_execution_status" in verifier
    assert "DEEPSEEK_API_KEY" not in verifier
    assert "OPENAI_API_KEY" not in verifier


def test_s9_verifier_treats_expected_test_lock_failure_as_evidence() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_s9.ps1").read_text(
        encoding="utf-8"
    )

    assert '$ErrorActionPreference = "Continue"' in verifier
    assert "$TestLockExitCode = $LASTEXITCODE" in verifier
    assert '$TestLockText -notmatch "S12"' in verifier
    assert '$Wheels.Count -ne 1 -or $SourceDistributions.Count -ne 1' in verifier


def test_s10_verifier_runs_frontend_gate_and_preserves_test_lock() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_s10.ps1").read_text(
        encoding="utf-8"
    )

    required_steps = (
        "scripts\\verify_s9.ps1",
        "npm ci",
        "npm test -- --run",
        "npm run typecheck",
        "npm run build",
        "locked_until_s12",
        "git diff --check",
        "S10_VERIFY_PASS",
    )
    positions = [verifier.index(step) for step in required_steps]
    assert positions == sorted(positions)
    assert 'Set-Location -LiteralPath (Join-Path $ProjectRoot "frontend")' in verifier
    assert "$env:OPENAI_API_KEY" not in verifier
    assert '"sk-[A-Za-z0-9_-]{20,}"' in verifier
    assert "Select-String -Pattern $CredentialPatterns" in verifier


def test_m0_verifier_safely_removes_only_frozen_tree_interpreter_caches() -> None:
    verifier = (PROJECT_ROOT / "scripts" / "verify_m0.ps1").read_text(
        encoding="utf-8"
    )

    assert "function Remove-FrozenInterpreterCache" in verifier
    assert "upstream_source\\packages\\ragbits-core" in verifier
    assert '$_ .Name -eq "__pycache__"'.replace(" ", "") in verifier.replace(" ", "")
    assert 'if (-not $CachePath.StartsWith($ExpectedPrefix' in verifier
    assert '-or\n            $Cache.Name -ne "__pycache__")' in verifier
    assert "Remove-Item -LiteralPath $CachePath -Recurse -Force" in verifier
    assert verifier.index("Remove-FrozenInterpreterCache") < verifier.index(
        '".\\baseline\\ragbits-reproduction\\VERIFY_PACKAGE.ps1"'
    )
    assert "function Remove-FrozenRuffCache" in verifier
    assert "baseline\\ragbits-reproduction\\upstream_source" in verifier
    assert '$Cache.Name -ne ".ruff_cache"' in verifier
    assert "Remove-FrozenRuffCache" in verifier


def test_s1_access_policy_record_covers_the_fixed_archive_hosts() -> None:
    policy = (PROJECT_ROOT / "data" / "ACCESS_POLICY.md").read_text(
        encoding="utf-8"
    )

    for expected in (
        "2026-10-14",
        "https://www.postgresql.org/robots.txt",
        "https://ftp.postgresql.org/robots.txt",
        "404 Not Found",
        "postgresql-17.11-docs.tar.gz",
        "postgresql-18.6-docs.tar.gz",
        "不进行网页抓取",
        "PostgreSQL License",
    ):
        assert expected in policy


def test_s7_readme_keeps_mock_credentials_and_test_boundaries_explicit() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    environment = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    for expected in (
        "S7 结构化生成与引用（已完成）",
        "Mock 仅验证工程契约",
        "OPENAI_API_KEY",
        "真实密钥不得提交",
        "Test 在 S12 前",
    ):
        assert expected in readme
    assert "gpt-5.4-mini-2026-03-17" in environment
    assert "sk-" not in environment


@pytest.fixture(scope="module")
def bundled_upstream_tree(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[str, str]]:
    clone_root = tmp_path_factory.mktemp("provenance") / "repository"
    bundle = BASELINE_ROOT / "provenance" / "ragbits-history.bundle"
    subprocess.run(
        ["git", "clone", "--no-checkout", "--quiet", str(bundle), str(clone_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = run_git("ls-tree", "-r", UPSTREAM_COMMIT, cwd=clone_root)
    return parse_tree_entries(output)


def test_archived_upstream_matches_fixed_bundle_tree(
    bundled_upstream_tree: dict[str, tuple[str, str]],
) -> None:
    outer_tree = parse_tree_entries(
        run_git("ls-files", "--stage", "--", UPSTREAM_PREFIX.rstrip("/")),
        prefix=UPSTREAM_PREFIX,
    )
    archived_files = {
        PurePosixPath(path.relative_to(BASELINE_ROOT / "upstream_source")).as_posix(): path
        for path in (BASELINE_ROOT / "upstream_source").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".ruff_cache" not in path.parts
    }

    assert len(bundled_upstream_tree) == 790
    assert set(outer_tree) == set(bundled_upstream_tree)
    assert set(archived_files) == set(bundled_upstream_tree)
    assert sum(mode == "100755" for mode, _ in bundled_upstream_tree.values()) == 10

    ordered_paths = sorted(bundled_upstream_tree)
    for path in ordered_paths:
        assert outer_tree[path] == bundled_upstream_tree[path]

    hash_result = subprocess.run(
        ["git", "hash-object", "--no-filters", "--stdin-paths"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        input="".join(f"{archived_files[path]}\n" for path in ordered_paths),
    )
    actual_blobs = hash_result.stdout.splitlines()
    expected_blobs = [bundled_upstream_tree[path][1] for path in ordered_paths]
    assert actual_blobs == expected_blobs
