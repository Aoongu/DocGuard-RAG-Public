import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import pytest

import docguard.ingestion.snapshot as snapshot_module
from docguard.ingestion.models import ArchiveSource, SnapshotRecord
from docguard.ingestion.snapshot import (
    ExtractionLimits,
    SnapshotConflictError,
    SnapshotError,
    UnsafeArchiveError,
    build_snapshot_records,
    inspect_archive,
    load_snapshot_manifest,
    main,
    restore_snapshot,
    verify_snapshot,
    write_snapshot_manifest,
    write_sources_manifest,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_archive(path: Path, version: str, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(content))


def _source(tmp_path: Path, version: str = "17.11") -> tuple[ArchiveSource, Path]:
    relative = Path("data/archives") / f"postgresql-{version}-docs.tar.gz"
    archive = tmp_path / relative
    files = {
        f"postgresql-{version}/doc/src/sgml/html/sql-select.html": b"select",
        f"postgresql-{version}/doc/src/sgml/html/backup-dump.html": b"backup",
        f"postgresql-{version}/doc/src/sgml/html/bookindex.html": b"ignored",
    }
    _write_archive(archive, version, files)
    source = ArchiveSource(
        version=version,
        major_version=version.split(".")[0],
        url=f"https://example.test/postgresql-{version}-docs.tar.gz",
        archive_path=PurePosixPath(relative.as_posix()),
        size_bytes=archive.stat().st_size,
        sha256=_sha(archive.read_bytes()),
        retrieved_at="2026-10-11T16:36:16Z",
    )
    return source, archive


def _records(source: ArchiveSource, archive: Path) -> tuple[SnapshotRecord, ...]:
    return build_snapshot_records(source, archive)


def test_build_records_selects_topics_and_sorts_stably(tmp_path: Path) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    assert [(item.archive_member.name, item.topic) for item in records] == [
        ("backup-dump.html", "backup_restore"),
        ("sql-select.html", "sql_commands"),
    ]
    assert records[0].content_sha256 == _sha(b"backup")


def test_restore_is_idempotent_and_versions_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source17, archive17 = _source(tmp_path, "17.11")
    source18, archive18 = _source(tmp_path, "18.6")
    records = _records(source17, archive17) + _records(source18, archive18)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"
    first = restore_snapshot((source17, source18), records, output)
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*.html")}
    second = restore_snapshot((source17, source18), records, output)
    assert first.restored == 4 and first.skipped == 0
    assert second.restored == 0 and second.skipped == 4
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*.html")} == before
    assert {p.name for p in output.iterdir() if p.name.startswith("postgresql-")} == {
        "postgresql-17.11",
        "postgresql-18.6",
    }


def test_conflict_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"
    restore_snapshot((source,), records, output)
    target = output / "postgresql-17.11/sql-select.html"
    target.write_bytes(b"changed")
    with pytest.raises(SnapshotConflictError, match="sql-select.html"):
        restore_snapshot((source,), records, output)
    assert target.read_bytes() == b"changed"


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape.html", "file"),
        ("/absolute.html", "file"),
        ("C:/drive.html", "file"),
        ("safe-link.html", "symlink"),
        ("safe-hard.html", "hardlink"),
        ("safe-fifo.html", "fifo"),
    ],
)
def test_unsafe_members_are_rejected(tmp_path: Path, name: str, kind: str) -> None:
    archive_path = tmp_path / "bad.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        if kind == "file":
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
        elif kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "target"
            archive.addfile(member)
        elif kind == "hardlink":
            member.type = tarfile.LNKTYPE
            member.linkname = "target"
            archive.addfile(member)
        else:
            member.type = tarfile.FIFOTYPE
            archive.addfile(member)
    with pytest.raises(UnsafeArchiveError):
        inspect_archive(archive_path, "17.11")


def test_wrong_archive_root_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "wrong.tar.gz"
    _write_archive(archive, "17.11", {"other/doc/src/sgml/html/sql-select.html": b"x"})
    with pytest.raises(UnsafeArchiveError, match="root"):
        inspect_archive(archive, "17.11")


@pytest.mark.parametrize(
    "limits",
    [
        ExtractionLimits(max_selected_members=1),
        ExtractionLimits(max_member_bytes=5),
        ExtractionLimits(max_total_bytes=5),
    ],
)
def test_resource_limits_are_enforced(tmp_path: Path, limits: ExtractionLimits) -> None:
    _, archive = _source(tmp_path)
    with pytest.raises(UnsafeArchiveError, match="limit"):
        inspect_archive(archive, "17.11", limits=limits)


def test_unselected_members_are_validated_but_never_read_or_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "mostly-ignored.tar.gz"
    root = "postgresql-17.11/doc/src/sgml/html"
    with tarfile.open(archive_path, "w:gz") as archive:
        directory = tarfile.TarInfo(root)
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for index in range(2_001):
            content = b"ignored"
            member = tarfile.TarInfo(f"postgresql-17.11/doc/ignored-{index}.txt")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        huge = b"x" * 64
        member = tarfile.TarInfo(f"{root}/bookindex.html")
        member.size = len(huge)
        archive.addfile(member, io.BytesIO(huge))
        selected = tarfile.TarInfo(f"{root}/sql-select.html")
        selected.size = 1
        archive.addfile(selected, io.BytesIO(b"s"))

    extracted: list[str] = []
    original = tarfile.TarFile.extractfile

    def recording_extractfile(
        archive: tarfile.TarFile, member: tarfile.TarInfo
    ) -> BinaryIO | None:
        extracted.append(member.name)
        return original(archive, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", recording_extractfile)
    result = inspect_archive(
        archive_path,
        "17.11",
        limits=ExtractionLimits(
            max_selected_members=1, max_member_bytes=8, max_total_bytes=8
        ),
    )
    assert [path.name for path in result] == ["sql-select.html"]
    assert extracted == [f"{root}/sql-select.html"]


def test_manifest_writers_are_stable_and_use_lf(tmp_path: Path) -> None:
    source, archive = _source(tmp_path)
    records = tuple(reversed(_records(source, archive)))
    sources_path = tmp_path / "sources.json"
    snapshot_path = tmp_path / "snapshot.jsonl"
    write_sources_manifest((source,), sources_path)
    write_snapshot_manifest(records, snapshot_path)
    assert sources_path.read_bytes().endswith(b"\n")
    assert snapshot_path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in sources_path.read_bytes() + snapshot_path.read_bytes()
    lines = snapshot_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["raw_path"] for line in lines] == sorted(
        json.loads(line)["raw_path"] for line in lines
    )
    assert load_snapshot_manifest(snapshot_path) == tuple(sorted(records, key=lambda r: str(r.raw_path)))
    source_payload = json.loads(sources_path.read_text(encoding="utf-8"))[0]
    assert source_payload["schema_version"] == 1
    assert source_payload["allowed_topics"]
    assert source_payload["selection_rules"]


@pytest.mark.parametrize("field", ["raw_path", "archive_member", "canonical_url"])
def test_duplicate_manifest_identity_is_rejected(tmp_path: Path, field: str) -> None:
    source, archive = _source(tmp_path)
    record = _records(source, archive)[0]
    path = tmp_path / "snapshot.jsonl"
    payload = _record_json(record)
    other = dict(payload)
    # An exact duplicate necessarily duplicates every identity; field documents the contract.
    path.write_text(json.dumps(payload) + "\n" + json.dumps(other) + "\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="duplicate"):
        load_snapshot_manifest(path)


def test_duplicate_json_property_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.jsonl"
    path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(SnapshotError, match="duplicate JSON key"):
        load_snapshot_manifest(path)


def _record_json(record: SnapshotRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "postgresql_version": record.postgresql_version,
        "topic": record.topic,
        "archive_path": str(record.archive_path),
        "archive_sha256": record.archive_sha256,
        "archive_member": str(record.archive_member),
        "raw_path": str(record.raw_path),
        "canonical_url": record.canonical_url,
        "content_sha256": record.content_sha256,
        "size_bytes": record.size_bytes,
        "media_type": record.media_type,
        "license": record.license,
    }


def test_missing_whitelisted_member_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    record = _records(source, archive)[0]
    missing = replace(
        record,
        archive_member=PurePosixPath(
            "postgresql-17.11/doc/src/sgml/html/backup-missing.html"
        ),
        raw_path=PurePosixPath("data/raw/postgresql-17.11/backup-missing.html"),
        canonical_url="https://www.postgresql.org/docs/17/backup-missing.html",
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SnapshotError, match="missing"):
        restore_snapshot((source,), (missing,), tmp_path / "raw")


def test_failed_version_restore_leaves_no_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = list(_records(source, archive))
    records[1] = replace(records[1], content_sha256="0" * 64)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"
    with pytest.raises(SnapshotError, match="hash"):
        restore_snapshot((source,), records, output)
    assert not (output / "postgresql-17.11").exists()


def test_stage_tampering_is_detected_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    original = snapshot_module._write_stage_file
    calls = 0

    def tampering_write(stage: Path, name: str, data: bytes) -> None:
        nonlocal calls
        original(stage, name, data)
        calls += 1
        if calls == len(records):
            (stage / name).write_bytes(b"short")

    monkeypatch.setattr(snapshot_module, "_write_stage_file", tampering_write)
    output = tmp_path / "raw"
    with pytest.raises(SnapshotError, match="staging"):
        restore_snapshot((source,), records, output)
    assert not (output / "postgresql-17.11").exists()


def test_short_stage_write_fails_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    original = os.write
    calls = 0

    def short_write(descriptor: int, data: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(descriptor, data[: max(1, len(data) // 2)])
        return 0

    monkeypatch.setattr(snapshot_module.os, "write", short_write)
    output = tmp_path / "raw"
    with pytest.raises(SnapshotError, match="short write"):
        restore_snapshot((source,), records, output)
    assert not (output / "postgresql-17.11").exists()


def test_stage_directory_replacement_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    original = snapshot_module._write_stage_file
    calls = 0

    def replacing_write(stage: Path, name: str, data: bytes) -> None:
        nonlocal calls
        original(stage, name, data)
        calls += 1
        if calls == len(records):
            moved = stage.with_name(stage.name + "-moved")
            stage.rename(moved)
            stage.mkdir()

    monkeypatch.setattr(snapshot_module, "_write_stage_file", replacing_write)
    output = tmp_path / "raw"
    with pytest.raises(SnapshotError, match="staging directory"):
        restore_snapshot((source,), records, output)
    assert not (output / "postgresql-17.11").exists()


def test_failed_stage_is_retained_without_path_cleanup_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    retained: dict[str, Path] = {}

    def replace_after_failure(stage: Path) -> None:
        original = stage.with_name(stage.name + "-original")
        stage.rename(original)
        stage.mkdir()
        (stage / "intruder.txt").write_bytes(b"intruder")
        retained["replacement"] = stage
        retained["original"] = original

    original_write = snapshot_module._write_stage_file
    calls = 0

    def fail_after_writes(stage: Path, name: str, data: bytes) -> None:
        nonlocal calls
        original_write(stage, name, data)
        calls += 1
        if calls == len(records):
            raise SnapshotError("injected staging failure")

    monkeypatch.setattr(snapshot_module, "_on_staging_failure", replace_after_failure)
    monkeypatch.setattr(snapshot_module, "_write_stage_file", fail_after_writes)
    output = tmp_path / "raw"
    with pytest.raises(SnapshotError, match="retained for manual inspection"):
        restore_snapshot((source,), records, output)
    assert (retained["replacement"] / "intruder.txt").read_bytes() == b"intruder"
    assert {path.name for path in retained["original"].iterdir()} == {
        "backup-dump.html",
        "sql-select.html",
    }
    assert not (output / "postgresql-17.11").exists()


def test_preexisting_stage_link_is_never_followed(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    victim = tmp_path / "victim.html"
    victim.write_bytes(b"victim")
    os.link(victim, stage / "sql-select.html")
    with pytest.raises(SnapshotError, match="staging file"):
        snapshot_module._write_stage_file(stage, "sql-select.html", b"replacement")
    assert victim.read_bytes() == b"victim"


def test_preexisting_stage_symlink_is_never_followed(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    victim = tmp_path / "victim.html"
    victim.write_bytes(b"victim")
    try:
        os.symlink(victim, stage / "sql-select.html")
    except OSError as error:
        pytest.skip(f"file symlink unavailable: {error}")
    with pytest.raises(SnapshotError, match="staging file"):
        snapshot_module._write_stage_file(stage, "sql-select.html", b"replacement")
    assert victim.read_bytes() == b"victim"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_preexisting_stage_reparse_point_is_never_followed(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    victim = tmp_path / "victim-dir"
    victim.mkdir()
    marker = victim / "marker.txt"
    marker.write_bytes(b"victim")
    target = stage / "sql-select.html"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(target), str(victim)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"junction unavailable: {result.stderr or result.stdout}")
    try:
        with pytest.raises(SnapshotError, match="staging file"):
            snapshot_module._write_stage_file(stage, target.name, b"replacement")
        assert marker.read_bytes() == b"victim"
    finally:
        if os.path.lexists(target):
            os.rmdir(target)


def test_target_appearing_after_precheck_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"

    def racing_publish(_stage: Path, destination: Path) -> None:
        destination.mkdir(parents=True)
        (destination / "intruder.html").write_bytes(b"intruder")

    monkeypatch.setattr(snapshot_module, "_before_atomic_publish", racing_publish)
    with pytest.raises(SnapshotConflictError):
        restore_snapshot((source,), records, output)
    assert (output / "postgresql-17.11/intruder.html").read_bytes() == b"intruder"


def test_atomic_publish_is_invisible_until_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"
    observed: list[set[str]] = []

    def observe(stage: Path, destination: Path) -> None:
        assert not os.path.lexists(destination)
        observed.append({path.name for path in stage.iterdir()})

    monkeypatch.setattr(snapshot_module, "_before_atomic_publish", observe)
    assert restore_snapshot((source,), records, output).restored == 2
    assert observed == [{"backup-dump.html", "sql-select.html"}]
    assert {path.name for path in (output / "postgresql-17.11").iterdir()} == observed[0]


@pytest.mark.parametrize("nonempty", [False, True])
def test_atomic_publish_never_replaces_racing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nonempty: bool
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"

    def race(_stage: Path, destination: Path) -> None:
        destination.mkdir()
        if nonempty:
            (destination / "intruder.html").write_bytes(b"intruder")

    monkeypatch.setattr(snapshot_module, "_before_atomic_publish", race)
    with pytest.raises(SnapshotConflictError):
        restore_snapshot((source,), records, output)
    destination = output / "postgresql-17.11"
    assert destination.is_dir()
    if nonempty:
        assert (destination / "intruder.html").read_bytes() == b"intruder"


def test_later_version_conflict_prevents_all_version_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source17, archive17 = _source(tmp_path, "17.11")
    source18, archive18 = _source(tmp_path, "18.6")
    records = _records(source17, archive17) + _records(source18, archive18)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"
    conflict = output / "postgresql-18.6"
    conflict.mkdir(parents=True)
    (conflict / "intruder.html").write_bytes(b"conflict")
    with pytest.raises(SnapshotConflictError):
        restore_snapshot((source17, source18), records, output)
    assert not (output / "postgresql-17.11").exists()


def test_second_version_publish_conflict_keeps_complete_first_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source17, archive17 = _source(tmp_path, "17.11")
    source18, archive18 = _source(tmp_path, "18.6")
    records = _records(source17, archive17) + _records(source18, archive18)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"

    def conflict_second(_stage: Path, destination: Path) -> None:
        if destination.name == "postgresql-18.6":
            destination.mkdir()
            (destination / "intruder.html").write_bytes(b"intruder")

    monkeypatch.setattr(snapshot_module, "_before_atomic_publish", conflict_second)
    with pytest.raises(SnapshotConflictError):
        restore_snapshot((source17, source18), records, output)
    verify_snapshot((source17,), _records(source17, archive17), output)
    assert (output / "postgresql-18.6/intruder.html").read_bytes() == b"intruder"


def test_external_replacement_after_first_publish_is_never_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source17, archive17 = _source(tmp_path, "17.11")
    source18, archive18 = _source(tmp_path, "18.6")
    records = _records(source17, archive17) + _records(source18, archive18)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"

    def replace_then_conflict(_stage: Path, destination: Path) -> None:
        if destination.name != "postgresql-18.6":
            return
        first = output / "postgresql-17.11"
        first.rename(output / "externally-displaced-17.11")
        first.mkdir()
        (first / "intruder.html").write_bytes(b"intruder-first")
        destination.mkdir()
        (destination / "intruder.html").write_bytes(b"intruder-second")

    monkeypatch.setattr(snapshot_module, "_before_atomic_publish", replace_then_conflict)
    with pytest.raises(SnapshotConflictError):
        restore_snapshot((source17, source18), records, output)
    assert (output / "postgresql-17.11/intruder.html").read_bytes() == b"intruder-first"
    assert (output / "postgresql-18.6/intruder.html").read_bytes() == b"intruder-second"


def test_existing_idempotent_version_is_never_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source17, archive17 = _source(tmp_path, "17.11")
    source18, archive18 = _source(tmp_path, "18.6")
    records17 = _records(source17, archive17)
    records = records17 + _records(source18, archive18)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"
    restore_snapshot((source17,), records17, output)
    before = {
        path.name: path.read_bytes()
        for path in (output / "postgresql-17.11").iterdir()
    }

    def fail_second(_stage: Path, destination: Path) -> None:
        if destination.name == "postgresql-18.6":
            raise OSError("injected publication failure")

    monkeypatch.setattr(snapshot_module, "_before_atomic_publish", fail_second)
    with pytest.raises(SnapshotError, match="staging retained for manual inspection"):
        restore_snapshot((source17, source18), records, output)
    assert {
        path.name: path.read_bytes()
        for path in (output / "postgresql-17.11").iterdir()
    } == before
    assert not (output / "postgresql-18.6").exists()


def test_output_root_file_symlink_is_rejected_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"victim")
    output = tmp_path / "raw-link"
    try:
        os.symlink(victim, output)
    except OSError as error:
        pytest.skip(f"file symlink unavailable: {error}")
    with pytest.raises(SnapshotError, match="output root"):
        restore_snapshot((source,), records, output)
    assert victim.read_bytes() == b"victim"


def test_output_root_directory_symlink_is_rejected_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "victim-dir"
    victim.mkdir()
    marker = victim / "marker.txt"
    marker.write_bytes(b"victim")
    output = tmp_path / "raw-link"
    try:
        os.symlink(victim, output, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")
    with pytest.raises(SnapshotError, match="output root"):
        restore_snapshot((source,), records, output)
    assert list(victim.iterdir()) == [marker]
    assert marker.read_bytes() == b"victim"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_output_root_junction_is_rejected_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "victim-dir"
    victim.mkdir()
    marker = victim / "marker.txt"
    marker.write_bytes(b"victim")
    output = tmp_path / "raw-junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(output), str(victim)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"junction unavailable: {result.stderr or result.stdout}")
    try:
        with pytest.raises(SnapshotError, match="output root"):
            restore_snapshot((source,), records, output)
        assert list(victim.iterdir()) == [marker]
        assert marker.read_bytes() == b"victim"
    finally:
        if os.path.lexists(output):
            os.rmdir(output)


def test_incomplete_manifest_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SnapshotError, match="whitelist mismatch"):
        restore_snapshot((source,), records[:1], tmp_path / "raw")


def test_verify_rejects_extra_managed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = _source(tmp_path)
    records = _records(source, archive)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "raw"
    restore_snapshot((source,), records, output)
    (output / "postgresql-17.11/extra.html").write_bytes(b"extra")
    with pytest.raises(SnapshotError, match="extra"):
        verify_snapshot((source,), records, output)


def test_cli_is_offline_by_default_and_download_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    called: list[Path] = []
    monkeypatch.setattr(
        "docguard.ingestion.snapshot.download_archive",
        lambda source, destination: called.append(destination),
    )
    assert main(["freeze"]) == 2
    assert called == []
    assert main(["freeze", "--download-missing"]) == 2
    assert called


def test_cli_freeze_downloads_only_when_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / "seed.tar.gz"
    _write_archive(
        seed,
        "17.11",
        {"postgresql-17.11/doc/src/sgml/html/sql-select.html": b"select"},
    )
    relative = PurePosixPath("data/archives/postgresql-17.11-docs.tar.gz")
    source = ArchiveSource(
        version="17.11",
        major_version="17",
        url="https://example.test/postgresql-17.11-docs.tar.gz",
        archive_path=relative,
        size_bytes=seed.stat().st_size,
        sha256=_sha(seed.read_bytes()),
        retrieved_at="2026-10-11T16:36:16Z",
    )
    calls: list[Path] = []

    def fake_download(_source: ArchiveSource, destination: Path) -> None:
        calls.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(seed, destination)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("docguard.ingestion.snapshot.SOURCES", (source,))
    monkeypatch.setattr("docguard.ingestion.snapshot.download_archive", fake_download)
    assert main(["freeze", "--download-missing"]) == 0
    assert calls == [Path(relative)]
    assert load_snapshot_manifest(Path("data/manifests/snapshot.jsonl"))


def test_declared_size_mismatch_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad-size.tar.gz"
    _write_archive(
        archive,
        "17.11",
        {"postgresql-17.11/doc/src/sgml/html/sql-select.html": b"payload" * 100},
    )
    archive.write_bytes(archive.read_bytes()[:-100])
    with pytest.raises(UnsafeArchiveError):
        inspect_archive(archive, "17.11")
