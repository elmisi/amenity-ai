from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

from .discovery import DiscoveryResult
from .scanner import ScanItem
from .task_state import TaskState

if TYPE_CHECKING:  # pragma: no cover
    from .settings import Settings


@dataclass(frozen=True)
class StatusCounts:
    total: int
    pending: int
    scanning: int
    scanned: int
    classifying: int
    classified: int
    moving: int
    moved: int
    skipped: int
    error: int


def count_statuses(items: list[ScanItem]) -> StatusCounts:
    pending = sum(1 for i in items if i.status == "pending")
    scanning = sum(1 for i in items if i.status == "scanning")
    scanned = sum(1 for i in items if i.status == "scanned")
    classifying = sum(1 for i in items if i.status == "classifying")
    classified = sum(1 for i in items if i.status == "classified")
    moving = sum(1 for i in items if i.status == "moving")
    moved = sum(1 for i in items if i.status == "moved")
    skipped = sum(1 for i in items if i.status == "skipped")
    error = sum(1 for i in items if i.status == "error")
    total = len(items)
    return StatusCounts(
        total=total,
        pending=pending,
        scanning=scanning,
        scanned=scanned,
        classifying=classifying,
        classified=classified,
        moving=moving,
        moved=moved,
        skipped=skipped,
        error=error,
    )


def derive_task_state(*, counts: StatusCounts, analysis: TaskState, scan: TaskState, archive: TaskState) -> str:
    state = "idle"
    if analysis.running:
        if analysis.cancel_requested:
            state = "stopping…"
        elif counts.classifying:
            state = "classifying…"
        elif counts.scanning:
            state = "scanning…"
        else:
            state = "running…"
    if scan.running:
        state = "scanning…"
    if archive.running:
        state = "archiving…"
    return state


def runtime_problem(
    discovery: DiscoveryResult | None, settings: "Settings"
) -> tuple[str | None, str]:
    """Severity of the current setup, judged by roles rather than by one provider.

    Reuses the doctor — same logic, two surfaces. `probe=None` keeps it pure:
    drawing a banner must never open a socket. Judging by role is what lets a
    provider stopped on purpose be a warning instead of an error, as long as
    something else answers for facts and vision.
    """
    if not discovery:
        return ("Detecting providers…", "info")

    from .doctor import STATUS_FAIL, STATUS_WARN, run_doctor

    report = run_doctor(discovery=discovery, settings=settings, probe=None)
    by_key = {c.key: c for c in report.checks}
    text = by_key.get("role.text")
    vision = by_key.get("role.vision")

    if text is not None and text.status == STATUS_FAIL:
        return ("No semantic model available", "error")
    if vision is not None and vision.status == STATUS_FAIL:
        return ("No vision model — images will be skipped", "warn")
    for check in report.checks:
        if check.key.startswith("provider.") and check.status == STATUS_FAIL:
            return (f"{check.key.split('.', 1)[1]} unreachable", "warn")
    for check in (text, vision):
        if check is not None and check.status == STATUS_WARN:
            return (check.detail or f"{check.label}: warning", "warn")
    return (None, "ok")


def banner_for_state(
    *,
    state: str,
    scanning: int,
    classifying: int,
    moving: int,
    problem: str | None,
    severity: str,
) -> tuple[str, str]:
    if severity == "error":
        base = problem or "Error"
        return (f"ERROR: {base}", "bold white on red")
    if severity == "info" and state == "idle":
        return ("Status: idle (detecting providers…)", "bold black on grey70")
    # A warning never replaces the running banner: it rides along with it, the
    # way `problem` already does in every branch below.
    if severity == "warn" and state == "idle":
        return (f"WARNING: {problem}", "bold black on yellow")
    if state == "idle":
        return ("Status: idle (no running task)", "bold black on grey70")
    if state.startswith("stopping"):
        in_flight = scanning + classifying + moving
        if in_flight:
            return (f"STOPPING — waiting for {in_flight} requests in flight",
                    "bold white on red")
        return ("STOPPING…", "bold white on red")
    if state.startswith("scanning") and scanning:
        msg = f"RUNNING: scanning — {scanning} in flight"
        if problem:
            msg += f" • {problem}"
        return (msg, "bold white on blue")
    if state.startswith("classifying") and classifying:
        msg = "RUNNING: classifying scanned files…"
        if problem:
            msg += f" • {problem}"
        return (msg, "bold white on blue")
    if state.startswith("archiving") and moving:
        msg = "RUNNING: moving files to archive…"
        if problem:
            msg += f" • {problem}"
        return (msg, "bold white on blue")
    if state.startswith("archiving"):
        msg = "RUNNING: archiving…"
        if problem:
            msg += f" • {problem}"
        return (msg, "bold white on blue")
    if state.startswith("scanning"):
        msg = "RUNNING: scanning directory…"
        if problem:
            msg += f" • {problem}"
        return (msg, "bold white on blue")
    msg = "RUNNING…"
    if problem:
        msg += f" • {problem}"
    return (msg, "bold white on blue")

