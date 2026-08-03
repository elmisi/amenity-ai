from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Mapping, Optional

from ..llm_router import generate_with_image_file


# Keywords that indicate a scanned document (in caption from vision model)
_DOCUMENT_KEYWORDS = frozenset([
    "document", "documento", "paper", "carta", "letter", "lettera",
    "invoice", "fattura", "receipt", "ricevuta", "bill", "bolletta",
    "contract", "contratto", "form", "modulo", "certificate", "certificato",
    "statement", "estratto", "report", "rapporto", "notice", "avviso",
    "text", "testo", "printed", "stampato", "typed", "dattiloscritto",
    "handwritten", "manoscritto", "scan", "scansione", "page", "pagina",
    "table", "tabella", "spreadsheet", "foglio", "official", "ufficiale",
    "signature", "firma", "stamp", "timbro", "header", "intestazione",
    "letterhead", "carta intestata", "memo", "memorandum", "pdf",
])


def _caption_indicates_document(caption: str) -> bool:
    """Check if the caption suggests a scanned document rather than a photo."""
    if not caption:
        return False
    caption_lower = caption.lower()
    # Check for document keywords
    for keyword in _DOCUMENT_KEYWORDS:
        if keyword in caption_lower:
            return True
    return False


@dataclass(frozen=True)
class ImageOcrMeta:
    ocr_time_s: float
    dpi: int
    variants: int


@dataclass(frozen=True)
class ImageCaptionMeta:
    caption_time_s: float
    vision_model_used: Optional[str] = None
    last_error: Optional[str] = None


def extract_image_text_ocr(
    path: Path,
    *,
    max_chars: int,
    ocr_mode: str,
) -> tuple[Optional[str], Optional[ImageOcrMeta]]:
    """Best-effort OCR for image files (jpg/png). Returns text + meta or (None, None)."""
    import shutil

    if not shutil.which("tesseract"):
        return None, None
    try:
        from PIL import Image, ImageOps
    except Exception:
        return None, None
    try:
        import pytesseract
    except Exception:
        return None, None

    def ocr_profile(mode: str) -> tuple[int, list[str], tuple[str, ...], bool, float]:
        m = (mode or "").lower()
        if m == "fast":
            return 220, ["6"], ("ita+eng", "eng"), False, 12.0
        if m == "high":
            return 300, ["3", "4", "6", "11"], ("ita+eng", "eng+ita", "eng"), True, 60.0
        return 260, ["6", "3"], ("ita+eng", "eng"), True, 25.0

    def score_text(text: str) -> float:
        t = (text or "").strip()
        if not t:
            return 0.0
        sample = t[:2000]
        alnum = sum(ch.isalnum() for ch in sample)
        letters = sum(ch.isalpha() for ch in sample)
        spaces = sum(ch.isspace() for ch in sample)
        weird = sum(ord(ch) < 9 or ord(ch) == 127 for ch in sample)
        artifacts = len(re.findall(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])_(?=[A-Za-zÀ-ÖØ-öø-ÿ])", sample))
        return (alnum + letters * 0.5 + spaces * 0.1) - (weird * 5.0 + artifacts * 3.0)

    def clean_ocr_artifacts(text: str) -> str:
        return re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])_(?=[A-Za-zÀ-ÖØ-öø-ÿ])", "", text or "")

    try:
        img = Image.open(path)
    except Exception:
        return None, None

    dpi, psms, langs, heavy_preprocess, budget_s = ocr_profile(ocr_mode)
    t0 = time.perf_counter()
    tesseract_config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"

    variants: list[Image.Image] = []
    try:
        variants.append(img)
        gray = img.convert("L")
        variants.append(gray)
        if heavy_preprocess:
            variants.append(ImageOps.autocontrast(gray))
            variants.append(ImageOps.autocontrast(gray).point(lambda x: 255 if x > 180 else 0, mode="1"))
    except Exception:
        variants = [img]

    best_text = ""
    best_score = 0.0
    for v in variants:
        for psm in psms:
            if time.perf_counter() - t0 > budget_s:
                break
            cfg = tesseract_config.replace("--psm 6", f"--psm {psm}")
            for lang in langs:
                if time.perf_counter() - t0 > budget_s:
                    break
                try:
                    text = pytesseract.image_to_string(v, lang=lang, config=cfg, timeout=30)
                except Exception:
                    continue
                text = clean_ocr_artifacts(text)
                s = score_text(text)
                if s > best_score:
                    best_score = s
                    best_text = text
            if time.perf_counter() - t0 > budget_s:
                break
        if time.perf_counter() - t0 > budget_s:
            break

    text = best_text.strip()
    if not text:
        return None, None
    # Avoid using very low-signal OCR (photos, noise).
    if best_score < 40.0 and len(text) < 120:
        return None, None
    ocr_elapsed = time.perf_counter() - t0
    return text[:max_chars], ImageOcrMeta(ocr_time_s=ocr_elapsed, dpi=dpi, variants=len(variants))


def caption_image(
    image_path: Path,
    *,
    vision_models: tuple[str, ...],
    prompt: str,
    provider_urls: Mapping[str, str],
    timeout_s: float = 180.0,
    max_retries: int = 1,
    limiter=None,
) -> tuple[str, ImageCaptionMeta]:
    """Try a list of vision models until one returns a caption (best-effort).

    Includes retry logic for timeout errors.
    """
    caption = ""
    used: str | None = None
    last_error: str | None = None
    t0 = time.perf_counter()
    for vm in vision_models:
        retries = 0
        while retries <= max_retries:
            try:
                cap = generate_with_image_file(
                    model=vm,
                    prompt=prompt,
                    image_path=str(image_path),
                    provider_urls=provider_urls,
                    timeout_s=timeout_s,
                    limiter=limiter,
                )
            except TimeoutError:
                retries += 1
                last_error = "TimeoutError"
                if retries <= max_retries:
                    continue  # Retry same model
                break  # Move to next model
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}"
                break  # Move to next model (don't retry non-timeout errors)

            if cap.error:
                # Check if error looks like a timeout
                if "timeout" in (cap.error or "").lower():
                    retries += 1
                    last_error = cap.error
                    if retries <= max_retries:
                        continue  # Retry
                last_error = cap.error
                break  # Move to next model

            caption = (cap.response or "").strip()
            if caption:
                used = vm
                break
            # Empty response - move to next model
            last_error = "Empty response"
            break

        if caption:
            break  # Success, exit outer loop

    elapsed = time.perf_counter() - t0
    return caption, ImageCaptionMeta(caption_time_s=elapsed, vision_model_used=used, last_error=last_error)


@dataclass(frozen=True)
class ImageExtractionResult:
    """Result of smart image extraction."""
    content: Optional[str]  # Text content (OCR text or caption for LLM)
    caption: Optional[str]  # Vision model caption
    is_document: bool  # Whether the image was detected as a document
    method: str  # "vision" or "vision+ocr"
    vision_time_s: float
    ocr_time_s: Optional[float]
    vision_model: Optional[str]
    ocr_meta: Optional[ImageOcrMeta]
    error: Optional[str]


def extract_image_smart(
    path: Path,
    *,
    vision_models: tuple[str, ...],
    vision_prompt: str,
    provider_urls: Mapping[str, str],
    ocr_mode: str,
    max_chars: int = 14000,
    vision_timeout_s: float = 120.0,
    limiter=None,
) -> ImageExtractionResult:
    """Smart image extraction: vision first, then OCR only if document detected.

    This optimized flow:
    1. Calls vision model (fast) to get a caption
    2. Analyzes caption to detect if it's a scanned document
    3. If document detected, runs OCR to extract text
    4. Returns appropriate content for text LLM

    For photos (non-documents), this skips the slow OCR step entirely.
    """
    # Step 1: Always call vision model first (fast)
    caption, cap_meta = caption_image(
        path,
        vision_models=vision_models,
        prompt=vision_prompt,
        provider_urls=provider_urls,
        timeout_s=vision_timeout_s,
        limiter=limiter,
    )

    if not caption:
        # Vision failed - try OCR as fallback (might be a scanned document)
        ocr_text, ocr_meta = extract_image_text_ocr(path, max_chars=max_chars, ocr_mode=ocr_mode)
        if ocr_text and ocr_meta:
            # OCR succeeded - use OCR content
            return ImageExtractionResult(
                content=ocr_text,
                caption=None,
                is_document=True,  # Assume document since OCR worked
                method="ocr",
                vision_time_s=cap_meta.caption_time_s,
                ocr_time_s=ocr_meta.ocr_time_s,
                vision_model=cap_meta.vision_model_used,
                ocr_meta=ocr_meta,
                error=None,
            )
        # Both vision and OCR failed
        vision_error = cap_meta.last_error or "Empty caption"
        return ImageExtractionResult(
            content=None,
            caption=None,
            is_document=False,
            method="vision",
            vision_time_s=cap_meta.caption_time_s,
            ocr_time_s=ocr_meta.ocr_time_s if ocr_meta else None,
            vision_model=cap_meta.vision_model_used,
            ocr_meta=ocr_meta,
            error=f"Vision failed ({vision_error}), OCR also failed",
        )

    # Step 2: Check if caption indicates a document
    is_document = _caption_indicates_document(caption)

    if is_document:
        # Step 3: Run OCR for documents
        ocr_text, ocr_meta = extract_image_text_ocr(path, max_chars=max_chars, ocr_mode=ocr_mode)
        if ocr_text and ocr_meta:
            # OCR successful - use OCR text as primary content
            return ImageExtractionResult(
                content=ocr_text,
                caption=caption,
                is_document=True,
                method="vision+ocr",
                vision_time_s=cap_meta.caption_time_s,
                ocr_time_s=ocr_meta.ocr_time_s,
                vision_model=cap_meta.vision_model_used,
                ocr_meta=ocr_meta,
                error=None,
            )
        # OCR failed - fall back to caption only
        return ImageExtractionResult(
            content=f"IMAGE_CAPTION: {caption}",
            caption=caption,
            is_document=True,
            method="vision",
            vision_time_s=cap_meta.caption_time_s,
            ocr_time_s=ocr_meta.ocr_time_s if ocr_meta else None,
            vision_model=cap_meta.vision_model_used,
            ocr_meta=ocr_meta,
            error="OCR failed, using caption only",
        )

    # Not a document - use caption only (skip OCR)
    return ImageExtractionResult(
        content=f"IMAGE_CAPTION: {caption}",
        caption=caption,
        is_document=False,
        method="vision",
        vision_time_s=cap_meta.caption_time_s,
        ocr_time_s=None,
        vision_model=cap_meta.vision_model_used,
        ocr_meta=None,
        error=None,
    )

