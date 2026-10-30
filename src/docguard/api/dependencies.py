"""Injected application services for the HTTP adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DependencyProbe = Callable[[], dict[str, tuple[bool, str | None]]]


@dataclass(frozen=True, slots=True)
class AppServices:
    query: Any
    artifacts: Any
    evaluations: Any
    dependency_probe: DependencyProbe
    shutdown: Any | None = None
    discovery: Any | None = None
