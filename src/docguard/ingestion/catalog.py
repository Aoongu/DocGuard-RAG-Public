"""Pinned PostgreSQL archives and ordered documentation topic rules."""

from collections.abc import Collection
from fnmatch import fnmatchcase
from pathlib import PurePosixPath

from docguard.ingestion.models import ArchiveSource, TopicRule

SOURCES = (
    ArchiveSource(
        version="17.11",
        major_version="17",
        url=(
            "https://ftp.postgresql.org/pub/source/v17.11/"
            "postgresql-17.11-docs.tar.gz"
        ),
        archive_path=PurePosixPath("data/archives/postgresql-17.11-docs.tar.gz"),
        size_bytes=3_843_559,
        sha256="cd1dbddebbd34ab038f6225a3f66411d1ed6e16b0e7f4b8cfa6f45e99642fa77",
        retrieved_at="2026-10-11T16:36:16Z",
    ),
    ArchiveSource(
        version="18.6",
        major_version="18",
        url=(
            "https://ftp.postgresql.org/pub/source/v18.6/"
            "postgresql-18.6-docs.tar.gz"
        ),
        archive_path=PurePosixPath("data/archives/postgresql-18.6-docs.tar.gz"),
        size_bytes=3_906_887,
        sha256="0419dec0d3b7ca55a80c0519a1dd88a8d172019ef9841288889f0281ea1f97ed",
        retrieved_at="2026-10-11T16:36:42Z",
    ),
)

TOPIC_RULES = (
    TopicRule(
        "auth_privileges",
        (
            "client-authentication*.html",
            "auth-*.html",
            "user-manag.html",
            "database-roles.html",
            "role-*.html",
            "ddl-priv.html",
            "ddl-rowsecurity.html",
            "sql-grant.html",
            "sql-revoke.html",
            "sql-createuser.html",
            "sql-alteruser.html",
            "sql-dropuser.html",
            "sql-createrole.html",
            "sql-alterrole.html",
            "sql-droprole.html",
            "sql-creategroup.html",
            "sql-altergroup.html",
            "sql-dropgroup.html",
            "sql-createpolicy.html",
            "sql-alterpolicy.html",
            "sql-droppolicy.html",
            "sql-reassign-owned.html",
            "sql-drop-owned.html",
        ),
    ),
    TopicRule(
        "backup_restore",
        (
            "backup*.html",
            "continuous-archiving.html",
            "app-pgdump.html",
            "app-pg-dumpall.html",
            "app-pgrestore.html",
            "app-pgbasebackup.html",
            "app-pgverifybackup.html",
            "app-pgcombinebackup.html",
            "app-pgreceivewal.html",
        ),
    ),
    TopicRule(
        "performance_tuning",
        (
            "performance-tips*.html",
            "using-explain*.html",
            "planner-*.html",
            "runtime-config-query.html",
            "runtime-config-resource.html",
            "runtime-config-statistics.html",
            "monitoring*.html",
            "indexes*.html",
            "parallel-query*.html",
            "pgstatstatements*.html",
            "auto-explain*.html",
        ),
    ),
    TopicRule(
        "installation_configuration",
        (
            "install-*.html",
            "installation*.html",
            "runtime.html",
            "runtime-config*.html",
            "config-setting.html",
            "server-start.html",
            "server-shutdown.html",
            "kernel-resources.html",
            "creating-cluster*.html",
            "app-initdb.html",
            "app-pg-ctl.html",
            "app-pgconfig.html",
        ),
    ),
    TopicRule(
        "sql_language",
        (
            "sql-syntax*.html",
            "ddl-*.html",
            "dml-*.html",
            "queries*.html",
            "datatype*.html",
            "functions*.html",
            "typeconv*.html",
            "textsearch*.html",
            "mvcc*.html",
        ),
    ),
    TopicRule("sql_commands", ("sql-*.html",)),
)


def assign_topics(names: Collection[str]) -> dict[str, str]:
    """Assign each matching file to its first matching topic rule."""
    assigned: dict[str, str] = {}
    for name in sorted(names):
        for rule in TOPIC_RULES:
            if any(fnmatchcase(name, pattern) for pattern in rule.patterns):
                assigned[name] = rule.name
                break
    return assigned
