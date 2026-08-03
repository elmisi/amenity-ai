from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from .textish import extract_textish_with_meta
from .types import TextExtractMeta
from .office import OfficeExtractMeta, extract_office_text_with_meta
from .pdf import PdfExtractMeta, extract_pdf_text_with_meta

ExtractMeta = Union[PdfExtractMeta, OfficeExtractMeta, TextExtractMeta]

# The kinds this registry can extract text from. The analyzer derives its
# dispatch from these instead of keeping a twin list: the two once drifted
# apart, and csv/html/yaml/gpx files were reported "Unsupported file type"
# while their extractors sat here unused.
OFFICE_KINDS: frozenset[str] = frozenset({"doc", "docx", "odt", "xls", "xlsx"})
TEXTISH_KINDS: frozenset[str] = frozenset(
    {"json", "md", "txt", "rtf", "svg", "kmz", "gpx", "html", "csv", "yaml"}
)
EXTRACTABLE_TEXT_KINDS: frozenset[str] = frozenset({"pdf"}) | OFFICE_KINDS | TEXTISH_KINDS


def extract_with_meta(
    *,
    kind: str,
    path: Path,
    max_chars: int = 15000,
    ocr_mode: str = "balanced",
) -> Tuple[Optional[str], Optional[str], Optional[ExtractMeta]]:
    """Single entrypoint for text extraction across supported kinds.

    Notes:
    - This is for the "scan/facts" phase. Image handling (OCR/vision) remains elsewhere.
    - Return shape matches the legacy per-module helpers: (text, reason, meta).
    """
    if kind == "pdf":
        return extract_pdf_text_with_meta(path, max_chars=max_chars, ocr_mode=ocr_mode)
    if kind in OFFICE_KINDS:
        return extract_office_text_with_meta(path, max_chars=max_chars)
    if kind in TEXTISH_KINDS:
        return extract_textish_with_meta(path, max_chars=max_chars)
    return None, "Unsupported file type", None
