"""Curated shortlist of installable models, by role.

Maintained the same way as CURATED_BIAS: by hand, now and then, in
dedicated passes. The sizes are indicative and exist so the choice is an
informed one before gigabytes land on a remote machine.
"""
from __future__ import annotations

from dataclasses import dataclass

_GB = 1024 ** 3


@dataclass(frozen=True)
class CatalogEntry:
    tag: str
    size_bytes: int
    note: str


_VISION = (
    CatalogEntry("moondream:latest", int(1.7 * _GB), "light and fast"),
    CatalogEntry("llava:7b", int(4.7 * _GB), "better quality"),
    CatalogEntry("minicpm-v:latest", int(5.5 * _GB), "strong on text inside images"),
)

_TEXT = (
    CatalogEntry("qwen2.5:3b-instruct", int(1.9 * _GB), "small and dependable"),
    CatalogEntry("qwen3:8b", int(5.2 * _GB), "a good compromise for classification"),
)


def catalog_for_role(role: str) -> tuple[CatalogEntry, ...]:
    return _VISION if role == "vision" else _TEXT
