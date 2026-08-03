from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Optional, Tuple
import zipfile


@dataclass(frozen=True)
class OfficeExtractMeta:
    method: str
    extract_time_s: float


def extract_office_text_with_meta(
    path: Path,
    *,
    max_chars: int = 15000,
) -> Tuple[Optional[str], Optional[str], Optional[OfficeExtractMeta]]:
    """Extract text from common Office formats.

    Supported (best-effort):
    - .docx: ZIP/XML parsing (no extra deps)
    - .xlsx: ZIP/XML parsing (no extra deps)
    - .odt, .ods: ZIP/XML parsing (no extra deps; same content.xml container)
    - .doc, .xls: requires LibreOffice (soffice) or antiword (doc only)
    """
    ext = path.suffix.lower().lstrip(".")
    t0 = time.perf_counter()

    if ext == "docx":
        text = _extract_docx_text(path, max_chars=max_chars)
        if not text:
            return None, "No extractable DOCX text", None
        return text, "docx", OfficeExtractMeta(method="docx", extract_time_s=time.perf_counter() - t0)

    if ext == "odt":
        text = _extract_odt_text(path, max_chars=max_chars)
        if not text:
            return None, "No extractable ODT text", None
        return text, "odt", OfficeExtractMeta(method="odt", extract_time_s=time.perf_counter() - t0)

    if ext == "ods":
        # Same container as odt: every ODF document keeps its text in
        # content.xml, and the extraction pulls text nodes generically.
        text = _extract_odt_text(path, max_chars=max_chars)
        if not text:
            return None, "No extractable ODS text", None
        return text, "ods", OfficeExtractMeta(method="ods", extract_time_s=time.perf_counter() - t0)

    if ext == "xlsx":
        text = _extract_xlsx_text(path, max_chars=max_chars)
        if not text:
            return None, "No extractable XLSX text", None
        return text, "xlsx", OfficeExtractMeta(method="xlsx", extract_time_s=time.perf_counter() - t0)

    if ext == "doc":
        text = _extract_doc_text_antiword(path, max_chars=max_chars)
        if text:
            return text, "antiword", OfficeExtractMeta(method="antiword", extract_time_s=time.perf_counter() - t0)
        text = _extract_via_libreoffice_txt(path, max_chars=max_chars)
        if text:
            return text, "libreoffice", OfficeExtractMeta(method="libreoffice", extract_time_s=time.perf_counter() - t0)
        return None, "No extractable DOC text (install libreoffice or antiword)", None

    if ext == "xls":
        text = _extract_via_libreoffice_txt(path, max_chars=max_chars)
        if text:
            return text, "libreoffice", OfficeExtractMeta(method="libreoffice", extract_time_s=time.perf_counter() - t0)
        return None, "No extractable XLS text (install libreoffice)", None

    return None, "Unsupported office type", None


def _extract_doc_text_antiword(path: Path, *, max_chars: int) -> Optional[str]:
    if not shutil.which("antiword"):
        return None
    try:
        proc = subprocess.run(
            ["antiword", str(path)],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:
        proc = None
    if not proc or proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    return text[:max_chars] if text else None


def _extract_via_libreoffice_txt(path: Path, *, max_chars: int) -> Optional[str]:
    if not shutil.which("soffice"):
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            proc = subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    str(outdir),
                    str(path),
                ],
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            if proc.returncode != 0:
                return None
            txt_files = list(outdir.glob("*.txt"))
            if not txt_files:
                return None
            text = txt_files[0].read_text(encoding="utf-8", errors="ignore").strip()
            return text[:max_chars] if text else None
    except Exception:
        return None


def _extract_docx_text(path: Path, *, max_chars: int) -> Optional[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception:
        return None
    text = _xml_to_text(xml)
    text = text.strip()
    return text[:max_chars] if text else None


def _extract_odt_text(path: Path, *, max_chars: int) -> Optional[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("content.xml").decode("utf-8", errors="ignore")
    except Exception:
        return None
    text = _xml_to_text(xml)
    text = text.strip()
    return text[:max_chars] if text else None


def _extract_xlsx_text(path: Path, *, max_chars: int) -> Optional[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            shared = ""
            try:
                shared = zf.read("xl/sharedStrings.xml").decode("utf-8", errors="ignore")
            except Exception:
                shared = ""

            sheet_texts: list[str] = []
            for name in sorted([n for n in zf.namelist() if n.startswith("xl/worksheets/") and n.endswith(".xml")])[:8]:
                try:
                    sheet_xml = zf.read(name).decode("utf-8", errors="ignore")
                except Exception:
                    continue
                sheet_texts.append(sheet_xml)

        # Very lightweight extraction: just grab text nodes and shared strings.
        chunks = []
        if shared:
            chunks.append(shared)
        chunks.extend(sheet_texts)
        xml = "\n".join(chunks)
    except Exception:
        return None

    text = _xml_to_text(xml)
    text = text.strip()
    return text[:max_chars] if text else None


def _xml_to_text(xml: str) -> str:
    # Avoid extra dependencies; pull visible text nodes.
    try:
        from xml.etree import ElementTree as ET

        root = ET.fromstring(xml)
        parts: list[str] = []
        for el in root.iter():
            if el.text:
                t = " ".join(el.text.split()).strip()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    except Exception:
        # Very rough fallback
        import re

        s = re.sub(r"<[^>]+>", " ", xml or "")
        return " ".join(s.split())

