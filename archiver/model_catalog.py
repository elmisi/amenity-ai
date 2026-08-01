"""Rosa curata dei modelli installabili, per ruolo.

Stesso criterio di manutenzione di CURATED_BIAS: a mano, ogni tanto, con
task dedicate. Le dimensioni sono indicative e servono a far scegliere
consapevolmente prima di scaricare gigabyte su una macchina remota.
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
    CatalogEntry("moondream:latest", int(1.7 * _GB), "leggero e veloce"),
    CatalogEntry("llava:7b", int(4.7 * _GB), "qualità migliore"),
    CatalogEntry("minicpm-v:latest", int(5.5 * _GB), "ottimo sul testo nelle immagini"),
)

_TEXT = (
    CatalogEntry("qwen2.5:3b-instruct", int(1.9 * _GB), "piccolo e affidabile"),
    CatalogEntry("qwen3:8b", int(5.2 * _GB), "buon compromesso per la classificazione"),
)


def catalog_for_role(role: str) -> tuple[CatalogEntry, ...]:
    return _VISION if role == "vision" else _TEXT
