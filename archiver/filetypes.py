from __future__ import annotations

from pathlib import Path
from typing import Optional


KIND_BY_EXTENSION: dict[str, str] = {
    # Documents
    "pdf": "pdf",
    # Images
    "jpg": "image",
    "jpeg": "image",
    "png": "image",
    "webp": "image",
    # Office
    "doc": "doc",
    "docx": "docx",
    "odt": "odt",
    "ods": "ods",
    "xls": "xls",
    "xlsx": "xlsx",
    "pptx": "pptx",
    # Email
    "eml": "eml",
    # Calendar
    "ics": "ics",
    # Text-ish
    "json": "json",
    "md": "md",
    "txt": "txt",
    "rtf": "rtf",
    "svg": "svg",
    "kmz": "kmz",
    # Data formats
    "csv": "csv",
    "yaml": "yaml",
    "yml": "yaml",
    # Web
    "html": "html",
    "htm": "html",
    # GPS
    "gpx": "gpx",
}


def infer_kind(path: Path) -> Optional[str]:
    ext = path.suffix.lower().lstrip(".")
    if not ext:
        return None
    return KIND_BY_EXTENSION.get(ext)

