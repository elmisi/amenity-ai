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


def provider_summary(discovery: "DiscoveryResult | None", settings: "Settings", *, model_picker) -> str:
    if not discovery:
        return ""
    provider = None
    models: tuple[str, ...] = ()
    for p in discovery.providers:
        if p.name == "ollama":
            provider = "ollama" if p.available else "ollama(missing)"
            models = p.models
            break
    if not provider:
        return ""

    text_models, vision_models = model_picker(discovery)
    facts = settings.facts_model if settings.facts_model and settings.facts_model != "auto" else (text_models[0] if text_models else "auto")
    classify = (
        settings.classify_model
        if settings.classify_model and settings.classify_model != "auto"
        else (text_models[0] if text_models else "auto")
    )
    if settings.vision_model and settings.vision_model != "auto":
        vision = settings.vision_model
    else:
        vision = vision_models[0] if vision_models else "auto"

    models_count = f"{len(models)} models" if models else "no models"
    return f"{provider} • {models_count} • facts={facts} • classify={classify} • vision={vision}"


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
