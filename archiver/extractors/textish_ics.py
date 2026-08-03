from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_MAX_EVENTS = 60
_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})(?:T(\d{2})(\d{2})\d{2}Z?)?$")


def _readable_date(value: str) -> str:
    """20240712T180000Z -> "2024-07-12 18:00".

    The dashes are not cosmetic: the analyzer's year-hint regex refuses a year
    followed by more digits, so "20240712" feeds it nothing and "2024-07-12"
    feeds it 2024.
    """
    m = _DATE.match(value.strip())
    if not m:
        return value.strip()
    out = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if m.group(4):
        out += f" {m.group(4)}:{m.group(5)}"
    return out


def _unescape(value: str) -> str:
    return (
        value.replace("\\n", " ").replace("\\N", " ")
        .replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")
    )


def extract_ics_text(path: Path, *, max_chars: int) -> Optional[str]:
    """Extract calendar events from an iCalendar file, stdlib only.

    One line per event: date, summary, location, then a trimmed description.
    Folded lines (RFC 5545: a continuation starts with a space or tab) are
    joined before parsing.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # Unfold: a line starting with space/tab continues the previous one.
    lines: list[str] = []
    for line in raw.splitlines():
        if line[:1] in (" ", "\t") and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)

    calendar_name = ""
    events: list[dict[str, str]] = []
    current: Optional[dict[str, str]] = None
    for line in lines:
        key, sep, value = line.partition(":")
        if not sep:
            continue
        name = key.split(";", 1)[0].upper()
        value = value.strip()
        if name == "X-WR-CALNAME":
            calendar_name = value
        elif name == "BEGIN" and value.upper() == "VEVENT":
            current = {}
        elif name == "END" and value.upper() == "VEVENT":
            if current:
                events.append(current)
            current = None
        elif current is not None and name in {"SUMMARY", "DTSTART", "DTEND", "LOCATION", "DESCRIPTION"}:
            current[name] = value

    if not events:
        return None

    parts: list[str] = []
    if calendar_name:
        parts.append(f"Calendar: {calendar_name}")
    parts.append(f"Events: {len(events)}")
    for event in events[:_MAX_EVENTS]:
        line = "- "
        if event.get("DTSTART"):
            line += _readable_date(event["DTSTART"]) + " "
        line += _unescape(event.get("SUMMARY", "(no title)"))
        if event.get("LOCATION"):
            line += f" @ {_unescape(event['LOCATION'])}"
        parts.append(line.strip())
        desc = _unescape(event.get("DESCRIPTION", "")).strip()
        if desc:
            if len(desc) > 200:
                desc = desc[:200] + "…"
            parts.append(f"  {desc}")
        if sum(len(p) for p in parts) >= max_chars:
            break

    out = "\n".join(parts).strip()
    return out[:max_chars] if out else None
