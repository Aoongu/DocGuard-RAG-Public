"""Map retrieval hits into deterministic, auditable S7 Evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from docguard.generation.models import Evidence, EvidenceProvenance

_CHUNK_ID_RE = re.compile(r"chk_[0-9a-f]{64}")
_VERSIONS = frozenset({"17.11", "18.6"})
_ROUTES = frozenset({"dense", "bm25"})


class EvidenceMappingError(RuntimeError):
    """Raised when a retrieval hit cannot become trustworthy Evidence."""


def _positive_rank(value: object, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise EvidenceMappingError(f"{field} must be a positive integer")
    return value


def _finite_score(value: object, field: str) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise EvidenceMappingError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EvidenceMappingError(f"{field} must be finite")
    return numeric


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise EvidenceMappingError(f"{field} must be non-empty text without NUL")
    return value


def _provenance(hit: Any, rank: int, score: float) -> EvidenceProvenance:
    raw_sources = getattr(hit, "sources", ("dense",))
    if (
        not isinstance(raw_sources, (tuple, list))
        or not raw_sources
        or any(item not in _ROUTES for item in raw_sources)
        or len(set(raw_sources)) != len(raw_sources)
    ):
        raise EvidenceMappingError("sources must contain unique dense/bm25 routes")
    sources = tuple(raw_sources)
    dense_rank = _positive_rank(getattr(hit, "dense_rank", rank), "dense_rank")
    dense_score = _finite_score(getattr(hit, "dense_score", score), "dense_score")
    bm25_rank = _positive_rank(getattr(hit, "bm25_rank", None), "bm25_rank")
    bm25_score = _finite_score(getattr(hit, "bm25_score", None), "bm25_score")
    rrf_score = _finite_score(getattr(hit, "rrf_score", None), "rrf_score")
    rerank_score = _finite_score(getattr(hit, "rerank_score", None), "rerank_score")
    original_rrf_rank = _positive_rank(
        getattr(hit, "original_rrf_rank", None), "original_rrf_rank"
    )
    rerank_truncated = getattr(hit, "rerank_truncated", None)
    if rerank_truncated is not None and type(rerank_truncated) is not bool:
        raise EvidenceMappingError("rerank_truncated must be boolean")
    return EvidenceProvenance(
        sources=sources,
        dense_rank=dense_rank if "dense" in sources else None,
        dense_score=dense_score if "dense" in sources else None,
        bm25_rank=bm25_rank if "bm25" in sources else None,
        bm25_score=bm25_score if "bm25" in sources else None,
        rrf_score=rrf_score,
        rerank_score=rerank_score,
        rerank_truncated=rerank_truncated,
        original_rrf_rank=original_rrf_rank,
    )


def build_evidence(
    hits: Sequence[Any],
    *,
    top_k: int,
    per_item_chars: int,
    total_chars: int,
) -> tuple[Evidence, ...]:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if type(per_item_chars) is not int or per_item_chars <= 0:
        raise ValueError("per_item_chars must be a positive integer")
    if type(total_chars) is not int or total_chars < top_k:
        raise ValueError("total_chars must provide at least one character per top_k item")

    selected = list(hits[:top_k])
    seen_chunks: set[str] = set()
    seen_ranks: set[int] = set()
    remaining = total_chars
    output: list[Evidence] = []
    for index, hit in enumerate(selected, start=1):
        rank = getattr(hit, "rank", None)
        if type(rank) is not int or rank <= 0:
            raise EvidenceMappingError("rank must be a positive integer")
        if rank in seen_ranks:
            raise EvidenceMappingError(f"duplicate rank: {rank}")
        if rank != index:
            raise EvidenceMappingError("rank must match retrieval order")
        seen_ranks.add(rank)

        chunk_id = getattr(hit, "chunk_id", None)
        if not isinstance(chunk_id, str) or _CHUNK_ID_RE.fullmatch(chunk_id) is None:
            raise EvidenceMappingError("chunk_id must be a full S2 chunk ID")
        if chunk_id in seen_chunks:
            raise EvidenceMappingError(f"duplicate chunk_id: {chunk_id}")
        seen_chunks.add(chunk_id)

        score = _finite_score(getattr(hit, "score", None), "score")
        if score is None:
            raise EvidenceMappingError("score must be finite")
        version = _string(getattr(hit, "postgresql_version", None), "postgresql_version")
        if version not in _VERSIONS:
            raise EvidenceMappingError("postgresql_version must be 17.11 or 18.6")
        topic = _string(getattr(hit, "topic", None), "topic")
        section_path = getattr(hit, "section_path", None)
        if (
            not isinstance(section_path, (tuple, list))
            or not section_path
            or not all(isinstance(item, str) and item.strip() for item in section_path)
        ):
            raise EvidenceMappingError("section_path must contain non-empty headings")
        canonical_url = _string(getattr(hit, "canonical_url", None), "canonical_url")
        parsed = urlsplit(canonical_url)
        major = version.split(".", 1)[0]
        if (
            parsed.scheme != "https"
            or parsed.netloc != "www.postgresql.org"
            or not parsed.path.startswith(f"/docs/{major}/")
            or parsed.query
            or parsed.fragment
        ):
            raise EvidenceMappingError("canonical_url must be an official versioned URL")
        anchor = _string(getattr(hit, "anchor", None), "anchor")
        text = _string(getattr(hit, "text", None), "text")

        slots = len(selected) - index + 1
        allocation = min(per_item_chars, remaining // slots)
        if allocation <= 0:
            raise EvidenceMappingError("total character budget exhausted")
        clipped = text[:allocation]
        remaining -= len(clipped)
        output.append(
            Evidence(
                evidence_id=f"E{index:02d}",
                chunk_id=chunk_id,
                rank=rank,
                score=score,
                postgresql_version=version,
                topic=topic,
                section_path=tuple(section_path),
                canonical_url=canonical_url,
                anchor=anchor,
                text=clipped,
                original_text_chars=len(text),
                text_truncated=len(clipped) < len(text),
                provenance=_provenance(hit, rank, score),
            )
        )
    return tuple(output)
