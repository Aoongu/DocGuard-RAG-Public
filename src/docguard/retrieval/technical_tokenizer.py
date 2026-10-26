"""Deterministic tokenizer for PostgreSQL terms, parameters, options, and errors."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(
    r"--[a-z0-9][a-z0-9_-]*|[a-z0-9]+(?:[_.-][a-z0-9]+)+|[a-z0-9]+"
)
_PART_RE = re.compile(r"[_.-]+")


def tokenize_technical(text: str) -> tuple[str, ...]:
    """Return stable full technical tokens followed by delimiter components."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    output: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0)
        output.append(token)
        body = token.removeprefix("--")
        if _PART_RE.search(body) is not None:
            output.extend(part for part in _PART_RE.split(body) if part)
    return tuple(output)
