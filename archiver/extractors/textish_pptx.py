from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def extract_pptx_text(path: Path, *, max_chars: int) -> Optional[str]:
    """Extract text from a .pptx: title, author, creation date, then slides.

    A pptx is a zip of XML, so the standard library is enough — same approach
    as kmz. Legacy binary .ppt is not handled here. Slide text lives in
    DrawingML `<a:t>` runs; document metadata in docProps/core.xml, whose
    creation date is exactly the year hint the analyzer feeds on.
    """
    import zipfile
    from xml.etree import ElementTree as ET

    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()

            slide_names: list[tuple[int, str]] = []
            for name in names:
                m = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
                if m:
                    slide_names.append((int(m.group(1)), name))
            slide_names.sort()

            parts: list[str] = []

            if "docProps/core.xml" in names:
                try:
                    core = ET.fromstring(zf.read("docProps/core.xml"))
                    for tag, label in (("title", "Title"), ("creator", "Author"), ("created", "Created")):
                        el = core.find(f".//{{*}}{tag}")
                        if el is not None and el.text and el.text.strip():
                            parts.append(f"{label}: {el.text.strip()}")
                except ET.ParseError:
                    pass

            for number, name in slide_names:
                try:
                    root = ET.fromstring(zf.read(name))
                except ET.ParseError:
                    continue
                runs = [
                    el.text.strip()
                    for el in root.iter()
                    if el.tag.endswith("}t") and el.text and el.text.strip()
                ]
                if runs:
                    parts.append(f"Slide {number}: " + " ".join(runs))
                if sum(len(p) for p in parts) >= max_chars:
                    break
    except Exception:
        return None

    out = "\n".join(parts).strip()
    return out[:max_chars] if out else None
