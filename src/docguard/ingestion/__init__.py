"""Versioned documentation ingestion primitives."""

from docguard.ingestion.catalog import SOURCES, TOPIC_RULES, assign_topics
from docguard.ingestion.models import ArchiveSource, SnapshotRecord, TopicRule

__all__ = [
    "SOURCES",
    "TOPIC_RULES",
    "ArchiveSource",
    "SnapshotRecord",
    "TopicRule",
    "assign_topics",
]
