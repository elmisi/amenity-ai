from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Optional, Sequence

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


@dataclass(frozen=True)
class RunProgress:
    """Progress of one run, not of the table.

    A table can hold files classified or moved in an earlier session; counting
    those as this run's progress is what produces numbers that do not add up.
    """

    total: int
    completed: int
    in_flight: int
    skipped: int = 0
    error: int = 0

    @property
    def queued(self) -> int:
        return max(0, self.total - self.completed - self.in_flight)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)


def compute_rate(
    timestamps: Sequence[float],
    *,
    now: float,
    started_at: float,
    window_s: float = 60.0,
) -> Optional[float]:
    """Completions per second over the trailing window.

    Divided by the time the window actually covers, not by the gap between the
    first and last completion in it. With several files in flight, completions
    arrive in bursts: measuring the gap between arrivals reads the burst and
    promises an ETA the run cannot keep.

    A sliding window rather than the whole run, so a slow stretch shows up
    instead of being diluted by everything that came before it.
    """
    window_start = max(started_at, now - window_s)
    covered = now - window_start
    if covered <= 0:
        return None
    recent = [t for t in timestamps if t >= window_start]
    if not recent:
        return None
    return len(recent) / covered


def format_eta(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return ""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"~{hours}h{minutes:02d}m left"
    if minutes:
        return f"~{minutes}m left"
    return f"~{secs}s left"


def progress_line(
    progress: RunProgress, *, rate: Optional[float], total_files: int
) -> str:
    bits = [
        f"files: {total_files}",
        f"queued: {progress.queued}",
        f"in flight: {progress.in_flight}",
        f"done: {progress.completed}",
        f"skipped: {progress.skipped}",
        f"error: {progress.error}",
    ]
    if rate:
        bits.append(f"{rate:.2f} file/s")
        eta = format_eta(progress.remaining / rate)
        if eta:
            bits.append(eta)
    return " • ".join(bits)


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
