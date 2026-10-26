"""Offline integration checks for the committed PostgreSQL snapshot."""

from __future__ import annotations

import json
import re
import socket
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from docguard.ingestion import snapshot
from docguard.ingestion.catalog import SOURCES, TOPIC_RULES, assign_topics
from docguard.ingestion.snapshot import (
    build_snapshot_records,
    load_snapshot_manifest,
    verify_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES_MANIFEST = PROJECT_ROOT / "data/manifests/sources.json"
SNAPSHOT_MANIFEST = PROJECT_ROOT / "data/manifests/snapshot.jsonl"


@pytest.fixture(autouse=True)
def forbid_public_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"integration test attempted network access: {args}, {kwargs}")

    def reject_download(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"offline CLI attempted archive download: {args}, {kwargs}")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(snapshot, "download_archive", reject_download)


def expected_sources_payload() -> list[dict[str, Any]]:
    allowed_topics = [rule.name for rule in TOPIC_RULES]
    selection_rules = [
        {"topic": rule.name, "patterns": list(rule.patterns)} for rule in TOPIC_RULES
    ]
    return [
        {
            "schema_version": 1,
            "version": source.version,
            "major_version": source.major_version,
            "url": source.url,
            "archive_path": str(source.archive_path),
            "size_bytes": source.size_bytes,
            "sha256": source.sha256,
            "retrieved_at": source.retrieved_at,
            "license_name": source.license_name,
            "license_url": source.license_url,
            "copyright": "PostgreSQL Global Development Group",
            "allowed_topics": allowed_topics,
            "selection_rules": selection_rules,
        }
        for source in SOURCES
    ]


def test_official_archives_resolve_to_the_frozen_manifest() -> None:
    records = load_snapshot_manifest(SNAPSHOT_MANIFEST)

    assert len(records) == 773
    assert Counter(record.postgresql_version for record in records) == {
        "17.11": 386,
        "18.6": 387,
    }
    assert (
        len(
            {
                (record.postgresql_version, record.archive_member)
                for record in records
            }
        )
        == 773
    )
    assert len({record.raw_path for record in records}) == 773
    assert len({record.canonical_url for record in records}) == 773

    for source in SOURCES:
        rebuilt = build_snapshot_records(source, PROJECT_ROOT / source.archive_path)
        frozen = tuple(
            record
            for record in records
            if record.postgresql_version == source.version
        )
        assert rebuilt == frozen

    verify_snapshot(SOURCES, records)


def test_all_sql_pages_and_topics_are_present_exactly_once() -> None:
    records = load_snapshot_manifest(SNAPSHOT_MANIFEST)
    allowed_topics = {rule.name for rule in TOPIC_RULES}

    assert {record.topic for record in records} == allowed_topics
    for source in SOURCES:
        version_records = [
            record
            for record in records
            if record.postgresql_version == source.version
        ]
        sql_records = [
            record
            for record in version_records
            if record.archive_member.name.startswith("sql-")
        ]
        assert len(sql_records) == 189
        assert len({record.archive_member for record in sql_records}) == 189
        assert all(
            assign_topics((record.archive_member.name,))[record.archive_member.name]
            == record.topic
            for record in version_records
        )
        assert all(
            record.canonical_url
            == (
                f"https://www.postgresql.org/docs/{source.major_version}/"
                f"{record.archive_member.name}"
            )
            for record in version_records
        )


def test_sources_manifest_matches_the_pinned_catalog() -> None:
    payload = json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8"))

    assert payload == expected_sources_payload()
    assert [item["retrieved_at"] for item in payload] == [
        "2026-10-11T16:36:16Z",
        "2026-10-11T16:36:42Z",
    ]
    assert [set(item) for item in payload] == [
        {
            "schema_version",
            "version",
            "major_version",
            "url",
            "archive_path",
            "size_bytes",
            "sha256",
            "retrieved_at",
            "license_name",
            "license_url",
            "copyright",
            "allowed_topics",
            "selection_rules",
        }
        for _ in SOURCES
    ]
    for item, source in zip(payload, SOURCES, strict=True):
        assert item["retrieved_at"] == source.retrieved_at
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            item["retrieved_at"],
        )


def test_snapshot_cli_restores_idempotently_and_verifies_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = load_snapshot_manifest(SNAPSHOT_MANIFEST)
    restore_results: list[snapshot.RestoreResult] = []
    real_restore = snapshot.restore_snapshot

    def record_restore(
        sources: Sequence[snapshot.ArchiveSource],
        frozen_records: Sequence[snapshot.SnapshotRecord],
        output_root: Path,
        *,
        limits: snapshot.ExtractionLimits = snapshot.ExtractionLimits(),  # noqa: B008
    ) -> snapshot.RestoreResult:
        result = real_restore(
            sources,
            frozen_records,
            output_root,
            limits=limits,
        )
        restore_results.append(result)
        return result

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setattr(snapshot, "restore_snapshot", record_restore)

    first_code = snapshot.main(["restore", "--output-root", str(tmp_path)])
    first_output = capsys.readouterr()

    assert first_code == 0
    assert first_output.out == ""
    assert first_output.err == ""
    assert (restore_results[-1].restored, restore_results[-1].skipped) == (773, 0)

    actual_files = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
        and not path.name.endswith(".restore.lock")
    }
    expected_files = {
        record.raw_path.relative_to("data/raw").as_posix() for record in records
    }
    assert actual_files == expected_files

    before_second = {
        path.relative_to(tmp_path).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    second_code = snapshot.main(["restore", "--output-root", str(tmp_path)])
    second_output = capsys.readouterr()
    after_second = {
        path.relative_to(tmp_path).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert second_code == 0
    assert second_output.out == ""
    assert second_output.err == ""
    assert (restore_results[-1].restored, restore_results[-1].skipped) == (0, 773)
    assert after_second == before_second

    verify_code = snapshot.main(["verify", "--output-root", str(tmp_path)])
    verify_output = capsys.readouterr()

    assert verify_code == 0
    assert verify_output.out == ""
    assert verify_output.err == ""
