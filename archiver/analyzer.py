from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional

from .llm_router import generate

from .extractors.image import extract_image_smart
from .extractors.registry import EXTRACTABLE_TEXT_KINDS, OFFICE_KINDS, extract_with_meta
from .scanner import ScanItem
from .taxonomy import DEFAULT_TAXONOMY_LINES, Taxonomy, parse_taxonomy_lines
from .utils_json import extract_json_dict
from .utils_parsing import (
    coerce_date_candidates,
    coerce_list,
    extract_amount_token,
    extract_date_token,
    is_year,
    name_token_count,
    short_entity,
    split_and_repair_tokens,
)
from .prompts import (
    build_facts_extraction_prompt,
    build_json_repair_prompt,
)

if TYPE_CHECKING:  # pragma: no cover
    from .concurrency import ConcurrencyLimiter

_DEFAULT_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)


@dataclass(frozen=True)
class AnalysisConfig:
    text_model: str = "gemma3:1b"
    vision_model: str = "moondream:latest"
    text_models: tuple[str, ...] = ()
    vision_models: tuple[str, ...] = ()
    provider_urls: dict[str, str] = None  # type: ignore[assignment]
    output_language: str = "auto"  # auto | it | en
    taxonomy: Taxonomy = _DEFAULT_TAXONOMY
    filename_separator: str = "space"  # space | underscore | dash
    ocr_mode: str = "balanced"  # fast | balanced | high
    # Absent when a single item is analysed on its own: with nothing else in
    # flight there is nothing to regulate.
    limiter: Optional["ConcurrencyLimiter"] = None

    def __post_init__(self) -> None:
        if self.provider_urls is None:
            object.__setattr__(self, "provider_urls", {})


@dataclass(frozen=True)
class FactsResult:
    status: str  # scanned | skipped | error
    reason: Optional[str] = None
    summary_long: Optional[str] = None
    facts_json: Optional[str] = None
    confidence: Optional[float] = None
    llm_raw_output: Optional[str] = None
    extract_method: Optional[str] = None
    extract_time_s: Optional[float] = None
    llm_time_s: Optional[float] = None
    ocr_time_s: Optional[float] = None
    ocr_mode: Optional[str] = None
    model_used: Optional[str] = None


# _is_year moved to utils_parsing.is_year


def _extract_year_hint_from_path(path: Path) -> Optional[str]:
    """Best-effort year extraction from filename/path.

    Examples it should catch:
    - 2021/...
    - ..._12.2019_...
    - 17.03.2020
    - 20200105_101112
    """

    text = " ".join([*path.parts, path.name])

    # Prefer a directory or token that is exactly a year.
    for part in reversed(path.parts):
        if is_year(part):
            return part

    # Month-year like mm.yyyy or mm-yyyy or mm_yyyy.
    m = re.search(r"(?<!\d)\d{1,2}[._-](19\d{2}|20\d{2})(?!\d)", text)
    if m:
        return m.group(1)

    # Dates like dd.mm.yyyy or dd-mm-yyyy or dd_mm_yyyy.
    m = re.search(r"(?<!\d)\d{1,2}[._-]\d{1,2}[._-](19\d{2}|20\d{2})(?!\d)", text)
    if m:
        return m.group(1)

    # Dates like dd-mm-yy / dd.mm.yy (2-digit year). Use a conservative pivot.
    m = re.search(r"(?<!\d)\d{1,2}[._-]\d{1,2}[._-](\d{2})(?!\d)", text)
    if m:
        yy = int(m.group(1))
        # 00-69 -> 2000-2069, 70-99 -> 1970-1999
        return str(2000 + yy) if yy <= 69 else str(1900 + yy)

    # ISO date yyyy-mm-dd.
    m = re.search(r"(?<!\d)(19\d{2}|20\d{2})-\d{1,2}-\d{1,2}(?!\d)", text)
    if m:
        return m.group(1)

    # Timestamps like yyyymmdd or yyyymmdd_hhmmss.
    m = re.search(r"(?<!\d)(19\d{2}|20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])(?!\d)", text)
    if m:
        return m.group(1)

    # Last resort: any year occurrence.
    m = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)
    if m:
        return m.group(1)

    return None


def _extract_year_hint_from_text(text: str) -> Optional[str]:
    # Keep it cheap and stable: first part of the document only.
    sample = text[:8000]
    years = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", sample)
    if not years:
        return None
    counts: dict[str, int] = {}
    for y in years:
        counts[y] = counts.get(y, 0) + 1
    # Prefer most frequent; tie-breaker: latest year.
    best = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[-1][0]
    return best


def _category_hint_from_signals(*, path: Path, text: Optional[str]) -> Optional[str]:
    hay = f"{path.as_posix()} {path.name}".lower()
    sample = (text or "")[:8000].lower()
    hay = hay + " " + sample

    # Home / utilities bills -> house.
    house_signals = [
        "bolletta",
        "fattura",
        "riepilogo fatture",
        "periodo riferimento",
        "protocollo",
        "gas naturale",
        "energia",
        "energia elettrica",
        "fornitura",
        "consumo",
        "kwh",
        "mc",
        "codice fiscale",
        "p.iva",
        "dolomiti energia",
        "enel",
        "iren",
        "acea",
        "acqua",
        "luce",
        "utenza",
    ]
    if any(s in hay for s in house_signals):
        return "house"

    # Technical docs / manuals -> tech.
    technical_signals = ["manual", "datasheet", "specification", "technical", "guide", "sdk", "api"]
    if any(s in hay for s in technical_signals):
        return "tech"

    return None


def _extract_json(text: str) -> Optional[dict]:
    return extract_json_dict(text)


_MAX_LLM_RAW_OUTPUT_CHARS = 12000
_JSON_RESPONSE_FORMAT = "json"
# num_predict is a ceiling, not a target: models that finish earlier cost nothing
# extra. These are sized so a mid-size model (7-8B) can emit a complete JSON object;
# the previous values were tuned for 1B models and truncated everything larger.
_JSON_REPAIR_OPTIONS = {"temperature": 0, "num_predict": 1500}
_FACTS_GENERATE_OPTIONS = {"temperature": 0, "num_predict": 2000}
_CLASSIFY_GENERATE_OPTIONS = {"temperature": 0, "num_predict": 1200}


def _truncate_raw_output(text: str) -> str:
    raw = (text or "").strip()
    if len(raw) <= _MAX_LLM_RAW_OUTPUT_CHARS:
        return raw
    return raw[: _MAX_LLM_RAW_OUTPUT_CHARS - 200] + "\n...[truncated]...\n" + raw[-200:]


def _repair_json_dict_via_llm(
    *,
    model: str,
    raw_output: str,
    provider_urls: Mapping[str, str],
    limiter: Optional["ConcurrencyLimiter"] = None,
) -> Optional[str]:
    snippet = _truncate_raw_output(raw_output)
    prompt = build_json_repair_prompt(snippet=snippet)
    try:
        gen = generate(
            model=model,
            prompt=prompt,
            provider_urls=provider_urls,
            timeout_s=60.0,
            response_format=_JSON_RESPONSE_FORMAT,
            think=False,
            keep_alive="5m",
            options=_JSON_REPAIR_OPTIONS,
            limiter=limiter,
        )
    except Exception:
        return None
    if gen.error:
        return None
    return gen.response


# The following functions have been moved to shared utilities:
# - _sanitize_name -> utils_filename.sanitize_name
# - _name_separator -> utils_filename.name_separator
# - _JOIN_BLOCKLIST -> utils_parsing.JOIN_BLOCKLIST
# - _split_and_repair_tokens -> utils_parsing.split_and_repair_tokens
# - _normalize_separators -> utils_filename.normalize_separators
# - _name_token_count -> utils_parsing.name_token_count
# - _coerce_list -> utils_parsing.coerce_list
# - _coerce_date_candidates -> utils_parsing.coerce_date_candidates


def _content_excerpt_for_llm(text: str, *, max_chars: int = 10000) -> str:
    """Keep within a predictable size while preserving high-signal regions.

    Strategy: if too long, keep head + tail (documents often contain totals/ids on the last part).
    """
    t = (text or "").strip()
    budget = max_chars
    # Very large documents can explode LLM latency; cap harder.
    if len(t) > max_chars * 8:
        budget = min(budget, 5000)
    elif len(t) > max_chars * 4:
        budget = min(budget, 8000)

    if len(t) <= budget:
        return t
    head = int(budget * 0.7)
    tail = budget - head
    return (t[:head].rstrip() + "\n\n…\n\n" + t[-tail:].lstrip()).strip()


# Additional functions moved to shared utilities:
# - _ensure_extension -> utils_filename.ensure_extension
# - _STOPWORDS -> utils_parsing.STOPWORDS
# - _GENERIC_NAME_TOKENS -> utils_parsing.GENERIC_NAME_TOKENS
# - _cleanup_generic_words_in_name -> utils_filename.cleanup_generic_words_in_name
# - _LEGAL_SUFFIXES_RE -> utils_parsing.LEGAL_SUFFIXES_RE
# - _short_entity -> utils_parsing.short_entity
# - _MONTHS -> utils_parsing.MONTHS
# - _extract_date_token -> utils_parsing.extract_date_token
# - _extract_amount_token -> utils_parsing.extract_amount_token
# - _propose_name_from_summary_and_facts -> utils_filename.propose_name_from_summary_and_facts


def _text_model_candidates(cfg: AnalysisConfig) -> tuple[str, ...]:
    if cfg.text_models:
        return cfg.text_models
    return (cfg.text_model,)


def _vision_model_candidates(cfg: AnalysisConfig) -> tuple[str, ...]:
    if cfg.vision_models:
        return cfg.vision_models
    return (cfg.vision_model,)


def _extract_facts_from_text(
    *,
    model: str,
    content: str,
    filename: str,
    mtime_iso: str,
    provider_urls: Mapping[str, str],
    year_hint_filename: Optional[str],
    year_hint_text: Optional[str],
    output_language: str,
    limiter: Optional["ConcurrencyLimiter"] = None,
) -> FactsResult:
    prompt = build_facts_extraction_prompt(
        filename=filename,
        mtime_iso=mtime_iso,
        year_hint_filename=year_hint_filename,
        year_hint_text=year_hint_text,
        content=content,
        output_language=output_language,
    )
    try:
        gen = generate(
            model=model,
            prompt=prompt,
            provider_urls=provider_urls,
            timeout_s=180.0,
            response_format=_JSON_RESPONSE_FORMAT,
            think=False,
            keep_alive="5m",
            options=_FACTS_GENERATE_OPTIONS,
            limiter=limiter,
        )
    except Exception as exc:  # noqa: BLE001
        return FactsResult(status="error", reason=f"LLM error: {type(exc).__name__}", model_used=model)
    if gen.error:
        return FactsResult(status="error", reason=f"LLM error: {gen.error}", model_used=model)

    out = gen.response
    data = _extract_json(out)
    llm_raw_output: Optional[str] = None
    if not isinstance(data, dict):
        llm_raw_output = _truncate_raw_output(out)
        repaired = _repair_json_dict_via_llm(
            model=model, raw_output=out, provider_urls=provider_urls, limiter=limiter
        )
        if repaired:
            data = _extract_json(repaired)
    if not isinstance(data, dict):
        return FactsResult(
            status="skipped",
            reason="Unparseable output (JSON)",
            model_used=model,
            llm_raw_output=llm_raw_output,
        )

    skip_reason = data.get("skip_reason")
    if isinstance(skip_reason, str) and skip_reason.strip():
        return FactsResult(status="skipped", reason=skip_reason.strip(), model_used=model)

    summary_long = data.get("summary_long")
    if not isinstance(summary_long, str) or not summary_long.strip():
        summary_long = None

    confidence = data.get("confidence")
    conf = float(confidence) if isinstance(confidence, (int, float)) else None
    if conf is not None and conf < 0.30:
        return FactsResult(
            status="skipped",
            reason="Low confidence",
            confidence=conf,
            model_used=model,
            llm_raw_output=llm_raw_output,
        )

    facts = {
        "language": data.get("language") if isinstance(data.get("language"), str) else None,
        "doc_type": data.get("doc_type") if isinstance(data.get("doc_type"), str) else None,
        "purpose": data.get("purpose") if isinstance(data.get("purpose"), str) else None,
        "tags": coerce_list(data.get("tags")),
        "people": coerce_list(data.get("people")),
        "organizations": coerce_list(data.get("organizations")),
        "addresses": coerce_list(data.get("addresses")),
        "amounts": data.get("amounts") if isinstance(data.get("amounts"), list) else [],
        "identifiers": data.get("identifiers") if isinstance(data.get("identifiers"), list) else [],
        "date_candidates": coerce_date_candidates(data.get("date_candidates")),
        "year_hint_filename": year_hint_filename,
        "year_hint_text": year_hint_text,
    }
    facts_json = json.dumps(facts, ensure_ascii=False, sort_keys=True)

    return FactsResult(
        status="scanned",
        summary_long=(summary_long or "").strip()[:4000] or None,
        facts_json=facts_json,
        confidence=conf,
        model_used=model,
        llm_raw_output=llm_raw_output,
    )


def extract_facts_item(item: ScanItem, *, config: AnalysisConfig) -> FactsResult:
    path = item.path
    if item.status not in {"pending", "skipped", "error"}:
        return FactsResult(status=item.status, reason=item.reason)

    year_hint_filename = _extract_year_hint_from_path(path)

    def skipped(reason: str) -> FactsResult:
        return FactsResult(status="skipped", reason=reason)

    if item.kind in EXTRACTABLE_TEXT_KINDS:
        text, reason, meta = extract_with_meta(kind=item.kind, path=path, ocr_mode=config.ocr_mode)
        if not text:
            if item.kind == "pdf":
                fallback = "No extractable PDF text"
            elif item.kind in OFFICE_KINDS:
                fallback = "No extractable office text"
            else:
                fallback = "No extractable text"

            res = skipped(reason or fallback)
            if meta:
                res = replace(
                    res,
                    extract_method=getattr(meta, "method", None),
                    extract_time_s=getattr(meta, "extract_time_s", None),
                    ocr_time_s=getattr(meta, "ocr_time_s", None),
                    ocr_mode=getattr(meta, "ocr_mode", None),
                )
            return res
        year_hint_text = _extract_year_hint_from_text(text)
        excerpt = _content_excerpt_for_llm(text, max_chars=10000)
        candidates = _text_model_candidates(config)
        t0 = time.perf_counter()
        res: Optional[FactsResult] = None
        for candidate_model in candidates:
            res = _extract_facts_from_text(
                model=candidate_model,
                content=excerpt,
                filename=path.name,
                mtime_iso=item.mtime_iso,
                provider_urls=config.provider_urls,
                year_hint_filename=year_hint_filename,
                year_hint_text=year_hint_text,
                output_language=config.output_language,
                limiter=config.limiter,
            )
            if res.status != "error":
                break
        llm_elapsed = time.perf_counter() - t0
        if meta:
            res = replace(
                res,
                extract_method=getattr(meta, "method", None),
                extract_time_s=getattr(meta, "extract_time_s", None),
                ocr_time_s=getattr(meta, "ocr_time_s", None),
                ocr_mode=getattr(meta, "ocr_mode", None),
            )
        return replace(res, llm_time_s=llm_elapsed)

    if item.kind == "image":
        max_chars = 10000
        # Smart extraction: vision first, OCR only if document detected
        if config.output_language == "it":
            vision_prompt = "Describe this image in one sentence in Italian."
        else:
            vision_prompt = "Describe this image in one sentence in English."

        img_result = extract_image_smart(
            path,
            vision_models=tuple(_vision_model_candidates(config)),
            vision_prompt=vision_prompt,
            provider_urls=config.provider_urls,
            ocr_mode=config.ocr_mode,
            max_chars=max_chars,
            limiter=config.limiter,
        )

        if not img_result.content:
            return skipped(img_result.error or "No content extracted from image")

        # Extract year hint from OCR text if available
        year_hint_text = None
        if img_result.method == "vision+ocr" and img_result.content and not img_result.content.startswith("IMAGE_CAPTION:"):
            year_hint_text = _extract_year_hint_from_text(img_result.content)

        content = img_result.content
        if img_result.method == "vision+ocr":
            content = _content_excerpt_for_llm(img_result.content, max_chars=max_chars)

        candidates = _text_model_candidates(config)
        model = candidates[0]
        t1 = time.perf_counter()
        res: Optional[FactsResult] = None
        for candidate_model in candidates:
            model = candidate_model
            res = _extract_facts_from_text(
                model=candidate_model,
                content=content,
                filename=path.name,
                mtime_iso=item.mtime_iso,
                provider_urls=config.provider_urls,
                year_hint_filename=year_hint_filename,
                year_hint_text=year_hint_text,
                output_language=config.output_language,
                limiter=config.limiter,
            )
            if res.status != "error":
                break
        llm_elapsed = time.perf_counter() - t1

        # Build model string
        used = model
        if img_result.vision_model:
            used = f"{model} (vision: {img_result.vision_model})"

        # Calculate total extract time
        extract_time = img_result.vision_time_s
        if img_result.ocr_time_s:
            extract_time += img_result.ocr_time_s

        return replace(
            res,
            extract_method=img_result.method,
            extract_time_s=extract_time,
            ocr_time_s=img_result.ocr_time_s,
            ocr_mode=config.ocr_mode if img_result.ocr_time_s else None,
            llm_time_s=llm_elapsed,
            model_used=used,
        )

    return skipped("Unsupported file type")


