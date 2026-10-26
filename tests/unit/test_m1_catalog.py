from dataclasses import FrozenInstanceError
from pathlib import PurePosixPath

import pytest

from docguard.ingestion.catalog import SOURCES, TOPIC_RULES, assign_topics
from docguard.ingestion.models import ArchiveSource, SnapshotRecord, TopicRule


def test_sources_pin_exact_official_archives() -> None:
    assert [
        (
            source.version,
            source.major_version,
            source.url,
            source.archive_path,
            source.size_bytes,
            source.sha256,
            source.retrieved_at,
            source.license_name,
            source.license_url,
        )
        for source in SOURCES
    ] == [
        (
            "17.11",
            "17",
            (
                "https://ftp.postgresql.org/pub/source/v17.11/"
                "postgresql-17.11-docs.tar.gz"
            ),
            PurePosixPath("data/archives/postgresql-17.11-docs.tar.gz"),
            3_843_559,
            "cd1dbddebbd34ab038f6225a3f66411d1ed6e16b0e7f4b8cfa6f45e99642fa77",
            "2026-10-11T16:36:16Z",
            "PostgreSQL",
            "https://www.postgresql.org/about/licence/",
        ),
        (
            "18.6",
            "18",
            (
                "https://ftp.postgresql.org/pub/source/v18.6/"
                "postgresql-18.6-docs.tar.gz"
            ),
            PurePosixPath("data/archives/postgresql-18.6-docs.tar.gz"),
            3_906_887,
            "0419dec0d3b7ca55a80c0519a1dd88a8d172019ef9841288889f0281ea1f97ed",
            "2026-10-11T16:36:42Z",
            "PostgreSQL",
            "https://www.postgresql.org/about/licence/",
        ),
    ]


def test_topic_rules_cover_six_technical_topics_in_priority_order() -> None:
    assert tuple(rule.name for rule in TOPIC_RULES) == (
        "auth_privileges",
        "backup_restore",
        "performance_tuning",
        "installation_configuration",
        "sql_language",
        "sql_commands",
    )


def test_topic_assignment_is_unique_and_uses_priority() -> None:
    assigned = assign_topics(
        {
            "auth-password.html",
            "runtime-config-query.html",
            "sql-grant.html",
            "sql-select.html",
            "queries-table-expressions.html",
            "unrelated.html",
        }
    )

    assert assigned == {
        "auth-password.html": "auth_privileges",
        "queries-table-expressions.html": "sql_language",
        "runtime-config-query.html": "performance_tuning",
        "sql-grant.html": "auth_privileges",
        "sql-select.html": "sql_commands",
    }


def test_models_are_frozen_and_slotted() -> None:
    source = SOURCES[0]

    with pytest.raises(FrozenInstanceError):
        source.version = "18.6"  # type: ignore[misc]
    assert not hasattr(source, "__dict__")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"version": "17"}, "version"),
        ({"major_version": "18"}, "major_version"),
        ({"url": "http://example.com/docs.tar.gz"}, "url"),
        ({"archive_path": PurePosixPath("/tmp/docs.tar.gz")}, "archive_path"),
        ({"archive_path": PurePosixPath("data/../docs.tar.gz")}, "archive_path"),
        ({"size_bytes": 0}, "size_bytes"),
        ({"sha256": "A" * 64}, "sha256"),
        ({"retrieved_at": "2026-10-11T16:36:16+00:00"}, "retrieved_at"),
    ],
)
def test_archive_source_rejects_invalid_inputs(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "version": "17.11",
        "major_version": "17",
        "url": "https://ftp.postgresql.org/pub/source/v17.11/"
        "postgresql-17.11-docs.tar.gz",
        "archive_path": PurePosixPath("data/archives/postgresql-17.11-docs.tar.gz"),
        "size_bytes": 1,
        "sha256": "0" * 64,
        "retrieved_at": "2026-10-11T16:36:16Z",
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError), match=message):
        ArchiveSource(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "url",
    (
        "http://localhost:8000/postgresql-docs.tar.gz",
        "http://127.0.0.1:8000/postgresql-docs.tar.gz",
        "http://[::1]:8000/postgresql-docs.tar.gz",
    ),
)
def test_archive_source_allows_http_only_for_local_test_servers(url: str) -> None:
    source = _archive_source(url=url)

    assert source.url == url


@pytest.mark.parametrize(
    "url",
    (
        "http://example.com/postgresql-docs.tar.gz",
        "http://user:password@localhost:8000/postgresql-docs.tar.gz",
        "https://user:password@ftp.postgresql.org/postgresql-docs.tar.gz",
    ),
)
def test_archive_source_rejects_nonlocal_http_and_url_credentials(url: str) -> None:
    with pytest.raises(ValueError, match="url"):
        _archive_source(url=url)


def test_topic_rule_rejects_empty_name_or_patterns() -> None:
    with pytest.raises(ValueError, match="name"):
        TopicRule("", ("*.html",))
    with pytest.raises(ValueError, match="patterns"):
        TopicRule("topic", ())


def test_snapshot_record_accepts_valid_design_fields() -> None:
    record = _snapshot_record()

    assert record.media_type == "text/html"
    assert record.license == "PostgreSQL"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": True}, "schema_version"),
        ({"postgresql_version": "16.9"}, "postgresql_version"),
        ({"topic": "other"}, "topic"),
        ({"archive_path": PurePosixPath("C:/archive.tar.gz")}, "archive_path"),
        (
            {
                "archive_path": PurePosixPath(
                    "data/archives/postgresql-18.6-docs.tar.gz"
                )
            },
            "archive_path",
        ),
        ({"archive_sha256": "bad"}, "archive_sha256"),
        ({"archive_member": PurePosixPath("../index.html")}, "archive_member"),
        (
            {
                "archive_member": PurePosixPath(
                    "postgresql-18.6/doc/src/sgml/html/sql-select.html"
                )
            },
            "archive_member",
        ),
        (
            {
                "archive_member": PurePosixPath(
                    "postgresql-17.11/doc/src/sgml/html/sql-grant.html"
                )
            },
            "basename",
        ),
        ({"raw_path": PurePosixPath("/data/raw/index.html")}, "raw_path"),
        ({"raw_path": PurePosixPath("data/raw/postgresql-18.6/index.html")}, "raw_path"),
        (
            {
                "raw_path": PurePosixPath(
                    "data/raw/postgresql-17.11/nested/sql-select.html"
                )
            },
            "raw_path",
        ),
        (
            {"canonical_url": "http://www.postgresql.org/docs/17/index.html"},
            "canonical_url",
        ),
        ({"canonical_url": "https://example.com/docs/17/index.html"}, "canonical_url"),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org:443/docs/17/sql-select.html"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://user:password@www.postgresql.org/docs/17/sql-select.html"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-select.html?download=1"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-select.html#syntax"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sub/../sql-select.html"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-grant.html"
                )
            },
            "basename",
        ),
        (
            {
                "archive_member": PurePosixPath(
                    "postgresql-17.11/doc/src/sgml/html/sql-select.HTML"
                ),
                "raw_path": PurePosixPath(
                    "data/raw/postgresql-17.11/sql-select.HTML"
                ),
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-select.HTML"
                ),
            },
            "basename",
        ),
        (
            {
                "archive_member": PurePosixPath(
                    "postgresql-17.11/doc/src/sgml/html/sql-select.htm"
                ),
                "raw_path": PurePosixPath(
                    "data/raw/postgresql-17.11/sql-select.htm"
                ),
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-select.htm"
                ),
            },
            "basename",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17//sql-select.html"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-select.html?"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-select.html#"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-\rselect.html"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-\nselect.html"
                )
            },
            "canonical_url",
        ),
        (
            {
                "canonical_url": (
                    "https://www.postgresql.org/docs/17/sql-\tselect.html"
                )
            },
            "canonical_url",
        ),
        ({"content_sha256": "F" * 64}, "content_sha256"),
        ({"size_bytes": -1}, "size_bytes"),
        ({"media_type": "text/plain"}, "media_type"),
        ({"license": "other"}, "license"),
    ],
)
def test_snapshot_record_rejects_invalid_inputs(
    changes: dict[str, object], message: str
) -> None:
    values = {
        "schema_version": 1,
        "postgresql_version": "17.11",
        "topic": "sql_commands",
        "archive_path": PurePosixPath("data/archives/postgresql-17.11-docs.tar.gz"),
        "archive_sha256": "0" * 64,
        "archive_member": PurePosixPath("postgresql-17.11/doc/src/sgml/html/sql-select.html"),
        "raw_path": PurePosixPath("data/raw/postgresql-17.11/sql-select.html"),
        "canonical_url": "https://www.postgresql.org/docs/17/sql-select.html",
        "content_sha256": "1" * 64,
        "size_bytes": 1,
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError), match=message):
        SnapshotRecord(**values)  # type: ignore[arg-type]


def _snapshot_record() -> SnapshotRecord:
    return SnapshotRecord(
        schema_version=1,
        postgresql_version="17.11",
        topic="sql_commands",
        archive_path=PurePosixPath("data/archives/postgresql-17.11-docs.tar.gz"),
        archive_sha256="0" * 64,
        archive_member=PurePosixPath(
            "postgresql-17.11/doc/src/sgml/html/sql-select.html"
        ),
        raw_path=PurePosixPath("data/raw/postgresql-17.11/sql-select.html"),
        canonical_url="https://www.postgresql.org/docs/17/sql-select.html",
        content_sha256="1" * 64,
        size_bytes=1,
    )


def _archive_source(*, url: str) -> ArchiveSource:
    return ArchiveSource(
        version="17.11",
        major_version="17",
        url=url,
        archive_path=PurePosixPath("data/archives/postgresql-17.11-docs.tar.gz"),
        size_bytes=1,
        sha256="0" * 64,
        retrieved_at="2026-10-11T16:36:16Z",
    )
