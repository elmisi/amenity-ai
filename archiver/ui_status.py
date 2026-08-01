from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING

from rich.text import Text

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import DiscoveryResult
    from .settings import Settings


def app_title(*, provider_line: str = "") -> str:
    try:
        ver = metadata.version("amenity-ai")
    except Exception:
        ver = "dev"
    base = f"amenity-ai v{ver}"
    if provider_line:
        return f"{base} • {provider_line}"
    return base


def status_cell(status: str) -> Text:
    # Backward-compatible mapping for older cache entries / statuses.
    status = {
        "analysis": "scanning",
        "extracting": "scanning",
        "extracted": "scanned",
        "ready": "classified",
        "normalizing": "classifying",
        "normalized": "classified",
    }.get(status, status)
    icon, style = {
        "pending": ("·", "bright_black"),
        "scanning": ("✓", "bright_blue"),
        "classifying": ("✓", "bright_blue"),
        "moving": ("✓", "bright_blue"),
        "scanned": ("✓", "yellow"),
        "classified": ("✓", "green"),
        "moved": ("✓", "cyan"),
        "skipped": ("✗", "red"),
        "error": ("✗", "red"),
    }.get(status, ("?", "bright_black"))
    return Text(icon, style=style)


def provider_summary(discovery: "DiscoveryResult | None", settings: "Settings") -> str:
    if not discovery:
        return ""
    from .model_selection import ROLE_CLASSIFY, ROLE_FACTS, ROLE_VISION, rank_models

    names = [s.name if s.available else f"{s.name}(down)"
             for s in discovery.providers if s.configured]
    if not names:
        return ""
    models = discovery.models

    def pick(pinned: str, role: str) -> str:
        if pinned and pinned != "auto":
            return pinned
        ranked = rank_models(models, role)
        return ranked[0] if ranked else "none"

    count = f"{len(models)} models" if models else "no models"
    return (
        f"{'+'.join(names)} • {count}"
        f" • facts={pick(settings.facts_model, ROLE_FACTS)}"
        f" • classify={pick(settings.classify_model, ROLE_CLASSIFY)}"
        f" • vision={pick(settings.vision_model, ROLE_VISION)}"
    )


def notes_line(
    *,
    scan_items_total: int,
    pending: int,
    scanning: int,
    scanned: int,
    classifying: int,
    classified: int,
    moved: int,
    skipped: int,
    error: int,
) -> str:
    bits = [
        f"files: {scan_items_total}" if scan_items_total else "files: 0",
        f"pending: {pending}",
        f"scanning: {scanning}",
        f"scanned: {scanned}",
        f"classifying: {classifying}",
        f"classified: {classified}",
        f"moved: {moved}",
        f"skipped: {skipped}",
        f"error: {error}",
    ]
    return " • ".join([b for b in bits if b])
