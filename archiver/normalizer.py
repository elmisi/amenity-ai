from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from .llm_router import generate
from .providers import split_model_id
from .scanner import ScanItem
from .taxonomy import Taxonomy, taxonomy_to_prompt_block
from .utils_filename import (
    ensure_extension,
    name_separator,
    normalize_separators,
    propose_name_from_summary_and_facts,
    sanitize_name,
)
from .utils_json import extract_json_any
from .utils_parsing import (
    GENERIC_NAME_TOKENS,
    STOPWORDS,
    is_year,
    name_token_count,
    short_entity,
    split_tokens,
    tokenize_for_match,
)
from .prompts import build_normalize_batch_prompt

def _normalize_options(batch_size: int) -> dict:
    """Token ceiling for one normalization call.

    Each item costs a JSON row (category, year, proposed name, summary), so the
    budget has to grow with the batch: a fixed 220 could not fit even two rows,
    which silently truncated every batch and forced the per-item fallback.
    """
    return {"temperature": 0, "num_predict": max(800, 300 * max(1, batch_size))}

_NORMALIZE_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "category": {"type": "string"},
            "reference_year": {"type": ["string", "null"]},
            "proposed_name": {"type": "string"},
            "summary": {"type": "string"},
            "confidence": {"type": ["number", "null"]},
        },
        "required": ["path", "category", "reference_year", "proposed_name", "summary", "confidence"],
    },
}


@dataclass(frozen=True)
class NormalizationResult:
    by_path: dict[str, dict]
    model_used: str
    error: Optional[str] = None


def _extract_json(text: str) -> Optional[object]:
    return extract_json_any(text)


def _chunk(items: list[ScanItem], size: int) -> list[list[ScanItem]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# The following functions/constants have been moved to shared utilities:
# - _sanitize_name -> utils_filename.sanitize_name
# - _name_separator -> utils_filename.name_separator
# - _normalize_separators -> utils_filename.normalize_separators
# - _ensure_extension -> utils_filename.ensure_extension
# - _GENERIC_NAME_TOKENS -> utils_parsing.GENERIC_NAME_TOKENS
# - _STOPWORDS -> utils_parsing.STOPWORDS
# - _LEGAL_SUFFIXES_RE -> utils_parsing.LEGAL_SUFFIXES_RE
# - _short_entity -> utils_parsing.short_entity
# - _tokenize_for_match -> utils_parsing.tokenize_for_match


def _category_repair_from_taxonomy(
    *,
    taxonomy: Taxonomy,
    summary_long: Optional[str],
    facts_obj: dict,
) -> Optional[str]:
    """Attempt to map content to a taxonomy category without an LLM.

    Uses user taxonomy examples/description as the weak supervision signal.
    Returns None if ambiguous / insufficient evidence.
    """

    # Build a small searchable text from already-extracted signals.
    fields: list[str] = []
    for key in ("doc_type",):
        v = facts_obj.get(key)
        if isinstance(v, str) and v.strip():
            fields.append(v)
    tags = facts_obj.get("tags")
    if isinstance(tags, list):
        fields.extend(str(t) for t in tags if isinstance(t, str))
    people = facts_obj.get("people")
    if isinstance(people, list) and people:
        # Avoid biasing toward personal for every named document; still useful for disambiguation.
        fields.append("people")
    orgs = facts_obj.get("organizations")
    if isinstance(orgs, list) and orgs:
        fields.append("organization")
    if isinstance(summary_long, str) and summary_long.strip():
        fields.append(summary_long)
    haystack = " ".join(fields)

    if not haystack.strip():
        return None

    tokens = set(tokenize_for_match(haystack))
    haystack_l = haystack.lower()

    scored: list[tuple[float, str]] = []
    for cat in taxonomy.categories:
        if cat.name == "unknown":
            continue
        kw_phrases: list[str] = []
        if cat.name:
            kw_phrases.append(cat.name)
        if cat.description:
            kw_phrases.append(cat.description)
        if cat.examples:
            kw_phrases.extend(cat.examples)

        score = 0.0
        for phrase in kw_phrases:
            if not phrase:
                continue
            p = phrase.strip().lower()
            if not p:
                continue
            # Phrase match is strong (e.g., "carta d'identità", "utility bill").
            if len(p) >= 6 and p in haystack_l:
                score += 3.0
                continue
            # Token match is weaker.
            for tok in tokenize_for_match(p):
                if tok in tokens:
                    score += 1.0

        if score > 0:
            scored.append((score, cat.name))

    if not scored:
        return None

    scored.sort(reverse=True)
    best_score, best_cat = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    # Require a minimum signal and avoid ties.
    if best_score < 3.0:
        return None
    if best_score - second_score < 1.0:
        return None
    return best_cat


# Additional functions moved to shared utilities:
# - _MONTHS -> utils_parsing.MONTHS
# - _extract_date_token -> utils_parsing.extract_date_token
# - _extract_amount_token -> utils_parsing.extract_amount_token
# - _name_token_count -> utils_parsing.name_token_count
# - _split_tokens -> utils_parsing.split_tokens


def _parse_facts_json(value: Optional[str]) -> dict:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        data = json.loads(value)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _compact_summary_long(value: Optional[str], *, max_chars: int = 280) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _facts_payload_for_model(facts_obj: dict) -> dict:
    if not isinstance(facts_obj, dict):
        return {}

    compact: dict[str, object] = {}
    for key in ("language", "doc_type", "purpose", "year_hint_filename", "year_hint_text"):
        value = facts_obj.get(key)
        if isinstance(value, str) and value.strip():
            compact[key] = value.strip()

    for key in ("tags", "people", "organizations", "addresses"):
        value = facts_obj.get(key)
        if isinstance(value, list):
            compact[key] = [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()][:4]

    amounts = facts_obj.get("amounts")
    if isinstance(amounts, list):
        compact["amounts"] = [item for item in amounts if isinstance(item, dict)][:3]

    identifiers = facts_obj.get("identifiers")
    if isinstance(identifiers, list):
        compact["identifiers"] = [item for item in identifiers if isinstance(item, dict)][:3]

    dates = facts_obj.get("date_candidates")
    if isinstance(dates, list):
        compact["date_candidates"] = [item for item in dates if isinstance(item, dict)][:4]

    return compact


def _best_year_from_facts(facts: dict, *, summary_long: Optional[str], proposed_name: Optional[str]) -> Optional[str]:
    # 1) explicit candidates from phase 1
    candidates = facts.get("date_candidates")
    best: tuple[float, str] | None = None
    if isinstance(candidates, list):
        for c in candidates:
            if not isinstance(c, dict):
                continue
            y = c.get("year")
            if not isinstance(y, str) or not re.fullmatch(r"(19\d{2}|20\d{2})", y.strip()):
                continue
            conf = c.get("confidence")
            score = float(conf) if isinstance(conf, (int, float)) else 0.0
            typ = c.get("type")
            if isinstance(typ, str) and typ.strip().lower() == "reference":
                score += 0.2
            if best is None or score > best[0]:
                best = (score, y.strip())
    if best is not None:
        return best[1]

    # 2) year hints collected in phase 1
    for key in ("year_hint_text", "year_hint_filename"):
        v = facts.get(key)
        if isinstance(v, str) and re.fullmatch(r"(19\d{2}|20\d{2})", v.strip()):
            return v.strip()

    # 3) extract from summary_long or proposed_name if present
    for text in (summary_long or "", proposed_name or ""):
        m = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)
        if m:
            return m.group(1)

    return None


def _propose_name_from_facts_json(
    *,
    summary_long: Optional[str],
    facts_json: Optional[str],
    reference_year: Optional[str],
    original_filename: str,
    filename_separator: str,
) -> Optional[str]:
    """Wrapper that parses facts_json before calling the shared implementation."""
    facts = _parse_facts_json(facts_json)
    return propose_name_from_summary_and_facts(
        summary_long=summary_long,
        facts=facts,
        reference_year=reference_year,
        original_filename=original_filename,
        filename_separator=filename_separator,
    )


def _normalize_chunk(
    batch: list[ScanItem],
    *,
    model: str,
    provider_urls: Mapping[str, str],
    taxonomy: Taxonomy,
    output_language: str,
    filename_separator: str,
    should_cancel: Optional[Callable[[], bool]],
    limiter,
    allowed,
    taxonomy_block: str,
    sep_desc: str,
    sep_label: str,
) -> tuple[dict, Optional[str]]:
    """Normalise one chunk and return (results, error).

    Independent of every other chunk: it never sees what the others produced.
    That independence is what lets chunks run at once, and it also repairs the
    single-item fallback at the end, which used to test a dict already filled
    by earlier chunks and so could never fire after the first one.
    """
    by_path: dict[str, dict] = {}
    payload = []
    by_input_path = {str(it.path): it for it in batch}
    token_to_path = {f"doc_{idx + 1}": str(it.path) for idx, it in enumerate(batch)}
    path_to_token = {path: token for token, path in token_to_path.items()}

    def fallback_to_single_items(error_reason: str) -> tuple[dict, Optional[str]]:
        fallback_by_path: dict[str, dict] = {}
        for single in batch:
            single_out, single_error = _normalize_chunk(
                [single],
                model=model,
                provider_urls=provider_urls,
                taxonomy=taxonomy,
                output_language=output_language,
                filename_separator=filename_separator,
                should_cancel=should_cancel,
                limiter=limiter,
                allowed=allowed,
                taxonomy_block=taxonomy_block,
                sep_desc=sep_desc,
                sep_label=sep_label,
            )
            if single_error:
                return fallback_by_path, error_reason
            fallback_by_path.update(single_out)
        return fallback_by_path, None

    def apply_row(row: dict, *, path: str) -> None:
        src = by_input_path.get(path)
        cat = row.get("category")
        if not isinstance(cat, str) or cat not in allowed:
            cat = "unknown"
        if cat == "unknown" and src:
            repaired = _category_repair_from_taxonomy(
                taxonomy=taxonomy,
                summary_long=src.summary_long,
                facts_obj=_parse_facts_json(src.facts_json),
            )
            if repaired in allowed:
                cat = repaired
        year = row.get("reference_year")
        if not isinstance(year, str) or not re.fullmatch(r"(19\\d{2}|20\\d{2})", year.strip()):
            year = None
        name = row.get("proposed_name")
        if not isinstance(name, str) or not name.strip():
            name = Path(path).name
        name = ensure_extension(sanitize_name(name.strip()), Path(path).name)
        name = normalize_separators(name, sep=sep_label)

        cur_facts = _parse_facts_json(src.facts_json) if src else {}

        derived_year = _best_year_from_facts(cur_facts, summary_long=src.summary_long, proposed_name=name) if src else None

        # If year is missing, derive it from facts/hints/summary.
        if not year and derived_year:
            year = derived_year
        # If the model picked a year that isn't evidenced, prefer the derived one.
        if year and derived_year and year != derived_year and src:
            evidence = f"{src.summary_long or ''} {name}"
            has_year = bool(re.search(rf"(?<!\d){re.escape(year)}(?!\d)", evidence))
            has_derived = bool(re.search(rf"(?<!\d){re.escape(derived_year)}(?!\d)", evidence))
            if (not has_year and has_derived) or (int(year) < 1950 and int(derived_year) >= 1950):
                year = derived_year

        # If the model output is generic, rebuild deterministically from summary_long + facts_json.
        if src and src.summary_long:
            orgs = cur_facts.get("organizations") if isinstance(cur_facts.get("organizations"), list) else []
            org_hint = short_entity(str(orgs[0])) if orgs else ""
            low_signal = len(Path(name).stem) < 18 or name_token_count(name) < 4
            missing_entity = bool(org_hint) and (org_hint.lower().split()[0] not in name.lower())
            if low_signal or missing_entity:
                better = _propose_name_from_facts_json(
                    summary_long=src.summary_long,
                    facts_json=src.facts_json,
                    reference_year=year,
                    original_filename=Path(path).name,
                    filename_separator=sep_label,
                )
                if better:
                    name = better

        summary = row.get("summary")
        if not isinstance(summary, str):
            summary = ""
        conf = row.get("confidence")
        conf_out = float(conf) if isinstance(conf, (int, float)) else None

        by_path[path] = {
            "category": cat,
            "reference_year": year,
            "proposed_name": name,
            "summary": summary.strip()[:200] or None,
            "confidence": conf_out,
            "model_used": model,
        }

    for it in batch:
        facts_obj = _parse_facts_json(it.facts_json)
        # Keep purpose in scan cache, but do not use it during classification/naming.
        if isinstance(facts_obj, dict) and "purpose" in facts_obj:
            facts_obj = dict(facts_obj)
            facts_obj.pop("purpose", None)
        path_str = str(it.path)
        payload.append(
            {
                "path": path_to_token.get(path_str, path_str),
                "kind": it.kind,
                "summary_long": _compact_summary_long(it.summary_long),
                "facts": _facts_payload_for_model(facts_obj),
                "current": {
                    "category": it.category,
                    "reference_year": it.reference_year,
                    "proposed_name": it.proposed_name,
                },
            }
        )

    prompt = build_normalize_batch_prompt(
        allowed_categories=list(allowed),
        taxonomy_block=taxonomy_block,
        separator_description=sep_desc,
        payload_json=json.dumps(payload, ensure_ascii=False),
        output_language=output_language,
    )

    if should_cancel and should_cancel():
        return by_path, "Cancelled"
    gen = generate(
        model=model,
        prompt=prompt,
        provider_urls=provider_urls,
        timeout_s=180.0,
        response_format=_NORMALIZE_RESPONSE_SCHEMA,
        think=False,
        keep_alive="5m",
        options=_normalize_options(len(batch)),
        limiter=limiter,
    )
    if gen.error:
        if len(batch) > 1:
            recovered, fallback_error = fallback_to_single_items(
                f"Batch normalization failed: {gen.error}"
            )
            if fallback_error is None:
                return recovered, None
        return by_path, gen.error
    data = _extract_json(gen.response)
    if isinstance(data, dict) and len(batch) == 1:
        data = [data]
    if not isinstance(data, list):
        if len(batch) > 1:
            recovered, fallback_error = fallback_to_single_items(
                "Unparseable output (JSON list)"
            )
            if fallback_error is None:
                return recovered, None
        return by_path, "Unparseable output (JSON list)"

    # Some models may fail to echo back the full absolute path. For single-item normalization we
    # can safely apply the first output row to the only input item.
    fallback_row: Optional[dict] = None
    for row in data:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        if not isinstance(path, str) or not path:
            if len(batch) == 1 and fallback_row is None:
                fallback_row = row
            continue
        actual_path = token_to_path.get(path, path)
        src = by_input_path.get(actual_path)
        # Ignore mismatched paths for multi-item batches; for per-file this is handled below.
        if not src:
            if len(batch) == 1 and fallback_row is None:
                fallback_row = row
            continue
        apply_row(row, path=actual_path)

    if len(batch) == 1 and not by_path and fallback_row is not None:
        only_path = str(batch[0].path)
        apply_row(fallback_row, path=only_path)

    return by_path, None


def normalize_items(
    *,
    items: list[ScanItem],
    model: str,
    taxonomy: Taxonomy,
    output_language: str,
    filename_separator: str,
    chunk_size: int = 25,
    should_cancel: Optional[Callable[[], bool]] = None,
    provider_urls: Optional[Mapping[str, str]] = None,
    limiter=None,
) -> NormalizationResult:
    """Normalise every item, one chunk of `chunk_size` per request.

    Chunks run concurrently up to the provider's own limit. Bigger chunks are
    not the alternative: a longer prompt is a longer request with more output
    and worse answers, and the chunk size was tuned.

    Each chunk is independent, results merge in input order, and the reported
    error is the first one raised by a chunk that produced nothing. A cancelled
    run says so regardless of what else went wrong.
    """
    provider_urls = provider_urls or {}
    allowed = taxonomy.allowed_names
    taxonomy_block = taxonomy_to_prompt_block(taxonomy)

    sep_label = filename_separator
    sep_desc = {
        "space": "spaces",
        "underscore": "underscores",
        "dash": "dashes",
    }.get(filename_separator, "spaces")

    chunks = _chunk(items, chunk_size)
    if not chunks:
        return NormalizationResult(by_path={}, model_used=model)

    def run(batch: list[ScanItem]) -> tuple[dict, Optional[str]]:
        return _normalize_chunk(
            batch,
            model=model,
            provider_urls=provider_urls,
            taxonomy=taxonomy,
            output_language=output_language,
            filename_separator=filename_separator,
            should_cancel=should_cancel,
            limiter=limiter,
            allowed=allowed,
            taxonomy_block=taxonomy_block,
            sep_desc=sep_desc,
            sep_label=sep_label,
        )

    workers = 1
    if limiter is not None:
        spec, _ = split_model_id(model)
        workers = max(1, min(len(chunks), limiter.limit(spec.name)))
    if workers == 1:
        outcomes = [run(batch) for batch in chunks]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="classify") as pool:
            outcomes = list(pool.map(run, chunks))

    by_path: dict[str, dict] = {}
    first_error: Optional[str] = None
    cancelled = False
    for partial, error in outcomes:
        by_path.update(partial)
        if error == "Cancelled":
            cancelled = True
        elif error and first_error is None:
            first_error = error
    if cancelled:
        return NormalizationResult(by_path=by_path, model_used=model, error="Cancelled")
    return NormalizationResult(by_path=by_path, model_used=model, error=first_error)



def normalize_items_with_fallback(
    *,
    items: list[ScanItem],
    models: tuple[str, ...],
    taxonomy: Taxonomy,
    output_language: str,
    filename_separator: str,
    chunk_size: int = 25,
    should_cancel: Optional[Callable[[], bool]] = None,
    provider_urls: Optional[Mapping[str, str]] = None,
    limiter=None,
) -> NormalizationResult:
    """Try each model in order; return the first result without an infra error.

    Cancellation is never retried. If every model errors, return the last result.
    """
    last: Optional[NormalizationResult] = None
    for model in models or ("",):
        result = normalize_items(
            items=items, model=model, provider_urls=provider_urls, taxonomy=taxonomy,
            output_language=output_language, filename_separator=filename_separator,
            chunk_size=chunk_size, should_cancel=should_cancel, limiter=limiter,
        )
        last = result
        if result.error is None or result.error == "Cancelled" or (should_cancel and should_cancel()):
            return result
    return last  # type: ignore[return-value]
