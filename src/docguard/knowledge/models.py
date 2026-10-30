"""Generic source records; unknown versions and missing anchors remain explicit."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Block:
    kind: Literal["paragraph", "list", "code", "table"]
    anchor: str | None
    section_path: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class Page:
    canonical_url: str
    title: str
    content_sha256: str
    blocks: tuple[Block, ...]
    links: tuple[str, ...]
    parser_version: str = "generic-html-v1"


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_revision: str
    knowledge_base_id: str
    product: str
    version: str | None
    canonical_url: str
    section_path: tuple[str, ...]
    anchor: str | None
    ordinal: int
    text: str
    token_count: int
    content_sha256: str
    parser_version: str
    chunking_version: str
