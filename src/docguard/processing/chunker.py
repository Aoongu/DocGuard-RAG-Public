"""Structure-aware deterministic chunking for normalized document blocks."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from docguard.processing.config import ChunkingConfig
from docguard.processing.models import (
    ChunkRecord,
    DocumentRecord,
    SectionBlock,
    make_chunk_id,
)
from docguard.processing.token_budget import TokenBudgetError

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;。！？；])\s+")


class ChunkingError(ValueError):
    """Raised when source structure cannot be represented within the budget."""


class Budget(Protocol):
    def count(self, text: str) -> int: ...

    def validate_chunk(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class _Fragment:
    text: str
    kind: str


def _render(section_path: tuple[str, ...], fragments: Sequence[_Fragment]) -> str:
    prefix = " > ".join(section_path)
    return prefix + "\n\n" + "\n\n".join(fragment.text for fragment in fragments)


def _fits(
    section_path: tuple[str, ...],
    fragments: Sequence[_Fragment],
    budget: Budget,
    maximum: int,
) -> bool:
    return budget.count(_render(section_path, fragments)) <= maximum


def _split_words(
    text: str,
    *,
    section_path: tuple[str, ...],
    budget: Budget,
    maximum: int,
    repeat: str | None = None,
    kind: str = "paragraph",
) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []

    def present(value: str) -> str:
        return f"{repeat}\n\n{value}" if repeat else value

    for word in words:
        candidate = " ".join((*current, word))
        if _fits(
            section_path,
            (_Fragment(present(candidate), kind),),
            budget,
            maximum,
        ):
            current.append(word)
            continue
        if current:
            pieces.append(present(" ".join(current)))
            current = []
        if _fits(
            section_path,
            (_Fragment(present(word), kind),),
            budget,
            maximum,
        ):
            current = [word]
            continue
        remaining = word
        while remaining:
            low, high, best = 1, len(remaining), 0
            while low <= high:
                middle = (low + high) // 2
                fragment = _Fragment(present(remaining[:middle]), kind)
                if _fits(section_path, (fragment,), budget, maximum):
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ChunkingError("token budget cannot represent a minimal text unit")
            pieces.append(present(remaining[:best]))
            remaining = remaining[best:]
    if current:
        pieces.append(present(" ".join(current)))
    return pieces


def _pack_text_units(
    units: Sequence[str],
    *,
    repeat: str | None,
    kind: str,
    section_path: tuple[str, ...],
    budget: Budget,
    maximum: int,
) -> list[_Fragment]:
    fragments: list[_Fragment] = []
    current: list[str] = []

    def body(values: Sequence[str]) -> str:
        content = "\n".join(values) if kind in {"table", "code", "list"} else " ".join(values)
        return f"{repeat}\n\n{content}" if repeat else content

    for unit in units:
        candidate = body((*current, unit))
        if _fits(section_path, (_Fragment(candidate, kind),), budget, maximum):
            current.append(unit)
            continue
        if current:
            fragments.append(_Fragment(body(current), kind))
            current = []
        single = body((unit,))
        if not _fits(section_path, (_Fragment(single, kind),), budget, maximum):
            if repeat is not None and kind != "table":
                raise ChunkingError(f"token budget cannot fit repeated {kind} context")
            split = _split_words(
                unit,
                section_path=section_path,
                budget=budget,
                maximum=maximum,
                repeat=repeat,
                kind=kind,
            )
            fragments.extend(_Fragment(piece, kind) for piece in split)
        else:
            current = [unit]
    if current:
        fragments.append(_Fragment(body(current), kind))
    return fragments


def _split_block(
    block: SectionBlock,
    *,
    explanation: str | None,
    budget: Budget,
    maximum: int,
) -> list[_Fragment]:
    whole = _Fragment(block.text, block.kind)
    if _fits(block.section_path, (whole,), budget, maximum):
        return [whole]
    if block.kind == "paragraph":
        sentences = [part for part in _SENTENCE_BOUNDARY.split(block.text) if part]
        expanded: list[str] = []
        for sentence in sentences:
            if _fits(
                block.section_path,
                (_Fragment(sentence, block.kind),),
                budget,
                maximum,
            ):
                expanded.append(sentence)
            else:
                expanded.extend(
                    _split_words(
                        sentence,
                        section_path=block.section_path,
                        budget=budget,
                        maximum=maximum,
                    )
                )
        return _pack_text_units(
            expanded,
            repeat=None,
            kind=block.kind,
            section_path=block.section_path,
            budget=budget,
            maximum=maximum,
        )
    lines = [line for line in block.text.splitlines() if line.strip()]
    if block.kind == "table":
        if len(lines) < 2:
            raise ChunkingError("token budget cannot fit a minimal table")
        return _pack_text_units(
            lines[1:],
            repeat=lines[0],
            kind=block.kind,
            section_path=block.section_path,
            budget=budget,
            maximum=maximum,
        )
    if block.kind == "code":
        return _pack_text_units(
            lines,
            repeat=explanation,
            kind=block.kind,
            section_path=block.section_path,
            budget=budget,
            maximum=maximum,
        )
    return _pack_text_units(
        lines,
        repeat=None,
        kind=block.kind,
        section_path=block.section_path,
        budget=budget,
        maximum=maximum,
    )


def _pack_fragments(
    fragments: Sequence[_Fragment],
    *,
    section_path: tuple[str, ...],
    config: ChunkingConfig,
    budget: Budget,
) -> list[list[_Fragment]]:
    packed: list[list[_Fragment]] = []
    current: list[_Fragment] = []
    for fragment in fragments:
        if _fits(
            section_path,
            (*current, fragment),
            budget,
            config.max_chunk_tokens,
        ):
            current.append(fragment)
            continue
        if not current:
            raise ChunkingError("fragment exceeds token budget after splitting")
        packed.append(current)
        overlap: list[_Fragment] = []
        for previous in reversed(current):
            candidate = [previous, *overlap]
            overlap_text = "\n\n".join(item.text for item in candidate)
            if budget.count(overlap_text) > config.overlap_tokens:
                break
            if not _fits(
                section_path,
                (*candidate, fragment),
                budget,
                config.max_chunk_tokens,
            ):
                break
            overlap = candidate
        current = [*overlap, fragment]
    if current:
        packed.append(current)
    if len(packed) >= 2:
        tail_text = _render(section_path, packed[-1])
        if budget.count(tail_text) < config.min_chunk_tokens:
            merged = [*packed[-2], *packed[-1]]
            if _fits(section_path, merged, budget, config.max_chunk_tokens):
                packed[-2:] = [merged]
    return packed


def chunk_document(
    document: DocumentRecord,
    blocks: Sequence[SectionBlock],
    config: ChunkingConfig,
    budget: Budget,
) -> tuple[ChunkRecord, ...]:
    """Convert ordered blocks to immutable chunks without crossing anchors."""
    if not blocks:
        raise ChunkingError("document must contain at least one block")
    segments: list[list[SectionBlock]] = []
    for block in blocks:
        if (
            not segments
            or segments[-1][-1].anchor != block.anchor
            or segments[-1][-1].section_path != block.section_path
        ):
            segments.append([block])
        else:
            segments[-1].append(block)

    ordinals: dict[str, int] = {}
    records: list[ChunkRecord] = []
    seen_ids: set[str] = set()
    for segment in segments:
        section_path = segment[0].section_path
        anchor = segment[0].anchor
        explanation: str | None = None
        fragments: list[_Fragment] = []
        for block in segment:
            fragments.extend(
                _split_block(
                    block,
                    explanation=explanation,
                    budget=budget,
                    maximum=config.max_chunk_tokens,
                )
            )
            if block.kind == "paragraph":
                explanation = block.text
        for packed in _pack_fragments(
            fragments,
            section_path=section_path,
            config=config,
            budget=budget,
        ):
            text = _render(section_path, packed)
            try:
                token_count = budget.validate_chunk(text)
            except TokenBudgetError as error:
                raise ChunkingError(f"{anchor}: token budget validation failed") from error
            if token_count > config.max_chunk_tokens:
                raise ChunkingError(f"{anchor}: token budget exceeds configured maximum")
            ordinal = ordinals.get(anchor, 0)
            chunk_id = make_chunk_id(
                config.chunking_version,
                document.postgresql_version,
                document.canonical_url,
                anchor,
                ordinal,
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
                    section_path=section_path,
                    ordinal=ordinal,
                    text=text,
                    token_count=token_count,
                    content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    chunking_version=config.chunking_version,
                )
            )
            ordinals[anchor] = ordinal + 1
    return tuple(records)
