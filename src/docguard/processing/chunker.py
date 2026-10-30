"""PostgreSQL chunk records backed by the shared structure-aware packer."""

import hashlib
from collections.abc import Sequence

from docguard.processing.block_packer import Budget, ChunkingError, pack_blocks
from docguard.processing.config import ChunkingConfig
from docguard.processing.models import (
    ChunkRecord,
    DocumentRecord,
    SectionBlock,
    make_chunk_id,
)

__all__ = ["Budget", "ChunkingError", "chunk_document"]


def chunk_document(
    document: DocumentRecord,
    blocks: Sequence[SectionBlock],
    config: ChunkingConfig,
    budget: Budget,
) -> tuple[ChunkRecord, ...]:
    records = []
    seen_ids = set()
    for packed in pack_blocks(blocks, config, budget):
        anchor = packed.anchor
        if anchor is None:
            raise ChunkingError("PostgreSQL blocks require anchors")
        chunk_id = make_chunk_id(
            config.chunking_version,
            document.postgresql_version,
            document.canonical_url,
            anchor,
            packed.ordinal,
        )
        if chunk_id in seen_ids:
            raise ChunkingError(f"duplicate chunk_id: {chunk_id}")
        seen_ids.add(chunk_id)
        records.append(
            ChunkRecord(
                schema_version=1,
                chunk_id=chunk_id,
                document_id=document.document_id,
                postgresql_version=document.postgresql_version,
                topic=document.topic,
                canonical_url=document.canonical_url,
                anchor=anchor,
                section_path=packed.section_path,
                ordinal=packed.ordinal,
                text=packed.text,
                token_count=packed.token_count,
                content_sha256=hashlib.sha256(packed.text.encode("utf-8")).hexdigest(),
                chunking_version=config.chunking_version,
            )
        )
    return tuple(records)
