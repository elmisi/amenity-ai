"""Utilities for filename manipulation and sanitization.

This module consolidates duplicated name manipulation code from analyzer.py
and normalizer.py.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .utils_parsing import (
    GENERIC_NAME_TOKENS,
    STOPWORDS,
    extract_amount_token,
    extract_date_token,
    is_year,
    short_entity,
    split_and_repair_tokens,
)


def sanitize_name(name: str, *, max_len: int = 180) -> str:
    """Remove invalid filename characters and normalize whitespace."""
    text = (name or "").strip()
    text = re.sub(r"[\\/:*?\"<>|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:max_len].strip()


def name_separator(kind: str) -> str:
    """Get the separator character for a given separator kind."""
    return {"space": " ", "underscore": "_", "dash": "-"}.get(kind, " ")


def normalize_separators(name: str, *, sep: str) -> str:
    """Normalize word separators in a filename to the desired separator.

    Uses split_and_repair_tokens for more robust token handling.
    """
    desired = name_separator(sep)
    stem = Path(name).stem
    ext = Path(name).suffix
    tokens = split_and_repair_tokens(stem)
    if not tokens:
        return sanitize_name(stem) + ext
    if desired == " ":
        return sanitize_name(" ".join(tokens)) + ext
    return sanitize_name(desired.join(tokens)) + ext


def ensure_extension(proposed_name: str, original_filename: str) -> str:
    """Ensure the proposed name has the same extension as the original."""
    original_ext = Path(original_filename).suffix
    if not original_ext:
        return proposed_name
    if proposed_name.lower().endswith(original_ext.lower()):
        return proposed_name
    return proposed_name.rstrip(".") + original_ext


def cleanup_generic_words_in_name(*, proposed_name: str, original_filename: str) -> str:
    """Remove generic words like 'this document' from model-proposed names.

    Keep meaningful document types (e.g. invoice, payslip) untouched.
    """
    original_ext = Path(original_filename).suffix
    ext = Path(proposed_name).suffix or original_ext
    stem = Path(proposed_name).stem
    tokens = split_and_repair_tokens(stem)
    cleaned: list[str] = []
    for t in tokens:
        if not t:
            continue
        if t.lower() in GENERIC_NAME_TOKENS:
            continue
        cleaned.append(t)
    if not cleaned:
        return sanitize_name(Path(original_filename).stem) + ext
    return sanitize_name(" ".join(cleaned)) + ext


def propose_name_from_summary_and_facts(
    *,
    summary_long: Optional[str],
    facts: dict,
    reference_year: Optional[str],
    original_filename: str,
    filename_separator: str,
) -> Optional[str]:
    """Generate a descriptive filename from summary and extracted facts.

    Returns None if insufficient information to generate a meaningful name.
    """
    if not summary_long:
        return None

    doc_type = facts.get("doc_type") if isinstance(facts.get("doc_type"), str) else ""
    tags = facts.get("tags") if isinstance(facts.get("tags"), list) else []
    doc_kind = doc_type.strip()
    if not doc_kind and tags:
        doc_kind = str(tags[0])
    doc_kind = re.sub(r"\b(document|documento|file|testo|immagine|image|text)\b", "", doc_kind, flags=re.IGNORECASE)
    doc_kind = re.sub(r"\s+", " ", doc_kind).strip()

    orgs = facts.get("organizations") if isinstance(facts.get("organizations"), list) else []
    people = facts.get("people") if isinstance(facts.get("people"), list) else []
    entity = ""
    if orgs:
        entity = short_entity(str(orgs[0]))
    elif people:
        entity = short_entity(str(people[0]))

    date_token = extract_date_token(summary_long) or (reference_year if reference_year and is_year(reference_year) else None)
    amount_token = extract_amount_token(summary_long)

    pieces: list[str] = []
    if doc_kind:
        pieces.extend(split_and_repair_tokens(doc_kind)[:4])
    if entity:
        pieces.extend(split_and_repair_tokens(entity)[:3])
    if date_token:
        pieces.append(date_token)
    if amount_token:
        pieces.extend(split_and_repair_tokens(amount_token)[:2])

    # De-duplicate while preserving order.
    seen: set[str] = set()
    cleaned: list[str] = []
    for p in pieces:
        pl = p.lower()
        if not pl or pl in STOPWORDS or pl in GENERIC_NAME_TOKENS:
            continue
        if pl in seen:
            continue
        seen.add(pl)
        cleaned.append(p)
        if len(cleaned) >= 10:
            break

    if len(cleaned) < 3:
        return None

    ext = Path(original_filename).suffix
    name = name_separator(filename_separator).join(cleaned)
    proposed = sanitize_name(name) + ext
    proposed = cleanup_generic_words_in_name(proposed_name=proposed, original_filename=original_filename)
    proposed = normalize_separators(proposed, sep=filename_separator)
    return proposed

