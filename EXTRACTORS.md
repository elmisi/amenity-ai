# Adding Support for New File Types

This guide explains how to add support for new file types in amenity-ai.

## Architecture Overview

The extractor system is modular and follows the **Open/Closed Principle**: you can add new file types without modifying existing code.

```
File → filetypes.py (extension → kind) → registry.py (kind → extractor) → Module
```

### Key Files

| File | Purpose |
|------|---------|
| `archiver/filetypes.py` | Maps file extensions to "kinds" |
| `archiver/extractors/registry.py` | Routes kinds to extractor functions |
| `archiver/extractors/types.py` | Common metadata dataclasses |
| `archiver/extractors/*.py` | Actual extraction logic |

## Step-by-Step: Adding a New File Type

### Example: Adding support for `.html` files

#### Step 1: Register the extension in `filetypes.py`

```python
KIND_BY_EXTENSION: dict[str, str] = {
    # ... existing ...
    "html": "html",  # Add this
    "htm": "html",   # Also add this alias
}
```

#### Step 2: Create the extractor module

Create `archiver/extractors/textish_html.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional


def extract_html_text(path: Path, *, max_chars: int = 15000) -> Optional[str]:
    """Extract visible text from an HTML file."""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # Simple approach: strip tags (or use BeautifulSoup if available)
    import re
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_chars] if text else None
```

#### Step 3: Add to registry

Update `archiver/extractors/registry.py`:

```python
def extract_with_meta(...) -> Tuple[Optional[str], Optional[str], Optional[ExtractMeta]]:
    # ... existing cases ...

    if kind == "html":
        from .textish_html import extract_html_text
        t0 = time.perf_counter()
        text = extract_html_text(path, max_chars=max_chars)
        if not text:
            return None, "No extractable HTML text", None
        from .types import TextExtractMeta
        return text, "html", TextExtractMeta(method="html", extract_time_s=time.perf_counter() - t0)
```

#### Step 4: Add to include_extensions (optional)

Update `archiver/settings.py` to include by default:

```python
include_extensions: tuple[str, ...] = (
    # ... existing ...
    "html",
    "htm",
)
```

#### Step 5: Test

```bash
# Refresh local install
~/.local/share/amenity-stuff/venv/bin/pip install -e .

# Test with a folder containing HTML files
python3 -m archiver --source /path/to/folder
```

## Extractor Function Signature

All extractors should follow this pattern:

```python
def extract_X_text(
    path: Path,
    *,
    max_chars: int = 15000,
) -> Optional[str]:
    """Extract text from X format.

    Returns:
        Extracted text (truncated to max_chars), or None if extraction failed.
    """
```

For formats that need OCR or have complex metadata, use the `_with_meta` variant:

```python
def extract_X_with_meta(
    path: Path,
    *,
    max_chars: int = 15000,
    ocr_mode: str = "balanced",  # optional
) -> Tuple[Optional[str], Optional[str], Optional[SomeMeta]]:
    """Extract text from X format with metadata.

    Returns:
        (text, method_or_skip_reason, metadata)
        - text: extracted content or None
        - method_or_skip_reason: extraction method used, or reason for skip
        - metadata: dataclass with timing and other info
    """
```

## Metadata Dataclasses

Define in `archiver/extractors/types.py` or in your module:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class HtmlExtractMeta:
    method: str  # e.g., "html", "html+beautifulsoup"
    extract_time_s: float
    # Add any format-specific fields
```

## Tips

1. **Use lazy imports**: Import heavy dependencies inside functions to avoid slow startup
2. **Handle errors gracefully**: Return `None` instead of raising exceptions
3. **Respect max_chars**: Always truncate output to avoid memory issues with large files
4. **Keep it simple**: Start with basic extraction, improve later if needed
5. **Test with edge cases**: Empty files, binary garbage, encoding issues

## Existing Extractors Reference

| Kind | Module | Dependencies |
|------|--------|--------------|
| `pdf` | `pdf.py` | pypdf, pymupdf, pytesseract (OCR) |
| `doc/docx/odt` | `office.py` | LibreOffice (via subprocess) |
| `xls/xlsx` | `office.py` | LibreOffice (via subprocess) |
| `image` | `image.py` | pytesseract, Pillow, Ollama vision |
| `txt/md` | `textish.py` | (none) |
| `json` | `textish.py` | (none) |
| `rtf` | `textish_rtf.py` | unrtf (optional) |
| `svg` | `textish_svg.py` | (none, uses xml.etree) |
| `kmz` | `textish_kmz.py` | (none, uses zipfile) |
| `gpx` | `textish_gpx.py` | (none, uses xml.etree) |
| `html/htm` | `textish_html.py` | (none, uses regex) |
| `csv` | `textish_csv.py` | (none, uses csv module) |
| `yaml/yml` | `textish_yaml.py` | (none) |
