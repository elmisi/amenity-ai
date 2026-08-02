"""Shared utilities for parsing dates, years, amounts, and text tokens.

This module consolidates duplicated code from analyzer.py and normalizer.py.
"""
from __future__ import annotations

import re
from typing import Optional


# ============================================================================
# Year validation and extraction
# ============================================================================

def is_year(value: str) -> bool:
    """Check if value is a valid year string (1900-2099)."""
    return bool(re.fullmatch(r"(19\d{2}|20\d{2})", value))


# ============================================================================
# Month names (Italian + English)
# ============================================================================

MONTHS: dict[str, int] = {
    # Italian
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12,
    # English
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


# ============================================================================
# Date extraction
# ============================================================================

def extract_date_token(text: str) -> Optional[str]:
    """Extract a date token (YYYY-MM-DD) from text.

    Supports multiple formats: ISO dates, European dates, dates with month names.
    Returns None if no valid date found.
    """
    t = text or ""

    # ISO format: yyyy-mm-dd
    m = re.search(r"(?<!\d)(19\d{2}|20\d{2})-(\d{1,2})-(\d{1,2})(?!\d)", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    # European format: dd.mm.yyyy or dd/mm/yyyy or dd-mm-yyyy
    m = re.search(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](19\d{2}|20\d{2})(?!\d)", t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    # Month name format: dd Month yyyy
    m = re.search(
        r"(?<!\d)(\d{1,2})\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)\s+(19\d{2}|20\d{2})(?!\d)",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        d = int(m.group(1))
        mon_name = m.group(2).strip().lower()
        y = int(m.group(3))
        mo = MONTHS.get(mon_name)
        if mo:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    return None


# ============================================================================
# Amount extraction
# ============================================================================

def extract_amount_token(text: str) -> Optional[str]:
    """Extract a currency amount from text.

    Supports formats like: "225,58 €", "€ 225.58", "225.58 EUR"
    Returns normalized format like "225.58 EUR" or None if not found.
    """
    t = text or ""

    # € followed by amount
    m = re.search(r"(€)\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,][0-9]{2})", t)
    if m:
        amount = m.group(2).replace(".", "").replace(",", ".")
        return f"{amount} EUR"

    # Amount followed by € or EUR/euro
    m = re.search(r"([0-9]{1,3}(?:[.,][0-9]{3})*[.,][0-9]{2})\s*(€|eur|euro)", t, flags=re.IGNORECASE)
    if m:
        amount = m.group(1).replace(".", "").replace(",", ".")
        return f"{amount} EUR"

    return None


# ============================================================================
# Stopwords and generic tokens
# ============================================================================

STOPWORDS: frozenset[str] = frozenset({
    # English
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "by", "from",
    # Italian
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "e", "o", "di", "da", "del", "della", "dei", "delle",
    "al", "alla", "alle", "agli", "per", "con", "su", "nel", "nella", "nelle", "all",
    # Generic / low-signal words often seen in LLM-generated names
    "this", "document", "doc", "file", "text", "image", "photo", "picture",
    "scan", "scanned", "documento", "immagine", "foto", "testo", "scansione",
})


GENERIC_NAME_TOKENS: frozenset[str] = frozenset({
    "this", "document", "doc", "file", "text", "image", "photo", "picture",
    "scan", "scanned", "documento", "immagine", "foto", "testo", "scansione",
})


# ============================================================================
# Legal suffixes for entity name cleanup
# ============================================================================

LEGAL_SUFFIXES_RE: re.Pattern[str] = re.compile(
    r"\b(s\.?p\.?a\.?|s\.?r\.?l\.?|srl|spa|inc\.?|llc|ltd\.?|gmbh|s\.?a\.?s\.?)\b",
    re.IGNORECASE,
)

# Tokens to drop after punctuation removal
LEGAL_SUFFIX_TOKENS: frozenset[str] = frozenset({
    "sp", "spa", "srl", "sa", "sas", "llc", "inc", "ltd", "gmbh"
})


# ============================================================================
# Token manipulation
# ============================================================================

# Words that should not be joined when repairing tokens. STOPWORDS already
# lists the function words of both languages, so keeping a second, shorter
# copy here only made the repair fire on the words it was missing.
JOIN_BLOCKLIST: frozenset[str] = STOPWORDS


def split_tokens(text: str) -> list[str]:
    """Split text into tokens, removing invalid filename characters."""
    raw = [t for t in re.split(r"[\s_\-]+", (text or "").strip()) if t]
    out: list[str] = []
    for t in raw:
        t2 = re.sub(r"[\\/:*?\"<>|]", " ", t).strip()
        if t2:
            out.append(t2)
    return out


def _looks_like_a_broken_word(a: str, b: str) -> bool:
    """True when `a` and `b` look like one word split by a stray separator.

    The artifact repaired here is "Mi_iti" for "Misiti": a short CAPITALISED
    fragment followed by a lowercase tail. Length alone does not identify it —
    "via roma", "tre spirali" and "per il" are short too, and gluing them
    corrupts the name for good. A capitalised head with a lowercase tail is
    the part that cannot be two ordinary words.
    """
    if not (a.isalpha() and b.isalpha()):
        return False
    if len(a) > 3 or a.lower() in JOIN_BLOCKLIST:
        return False
    return a[:1].isupper() and a[1:].islower() and b.islower()


def split_and_repair_tokens(stem: str) -> list[str]:
    """Split a filename stem into tokens and repair common OCR/encoding artifacts.

    Example artifact: "Mi_iti" -> ["Miiti"] (missing character, but keeps the word together)
    """
    tokens = split_tokens(stem)

    i = 0
    while i < len(tokens) - 1:
        if _looks_like_a_broken_word(tokens[i], tokens[i + 1]):
            tokens[i] = tokens[i] + tokens[i + 1]
            del tokens[i + 1]
            continue
        i += 1
    return tokens


def tokenize_for_match(text: str) -> list[str]:
    """Tokenize text for fuzzy matching (lowercase, cleaned, min length 3)."""
    t = (text or "").lower()
    t = re.sub(r"[^\w\sÀ-ÖØ-öø-ÿ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return [p for p in t.split() if p and (len(p) >= 3 or p.isdigit())]


def name_token_count(name: str) -> int:
    """Count the number of meaningful tokens in a filename stem."""
    from pathlib import Path
    return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", Path(name).stem))


# ============================================================================
# Entity name cleanup
# ============================================================================

def short_entity(entity: str) -> str:
    """Shorten an entity name by removing legal suffixes and keeping first 3 words."""
    e = re.sub(r"[^\w\sÀ-ÖØ-öø-ÿ]", " ", (entity or "")).strip()
    e = LEGAL_SUFFIXES_RE.sub("", e).strip()
    e = re.sub(r"\s+", " ", e)
    parts = [p for p in e.split() if len(p) >= 2]
    parts = [p for p in parts if p.lower() not in LEGAL_SUFFIX_TOKENS]
    return " ".join(parts[:3]).strip()


# ============================================================================
# Coercion helpers (for LLM output parsing)
# ============================================================================

def coerce_list(value: object) -> list[str]:
    """Coerce a value to a list of non-empty strings."""
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        return out
    return []


def coerce_date_candidates(value: object) -> list[dict]:
    """Coerce a value to a list of date candidate dicts."""
    out: list[dict] = []
    if not isinstance(value, list):
        return out
    for v in value:
        if not isinstance(v, dict):
            continue
        year = v.get("year")
        if isinstance(year, str) and is_year(year.strip()):
            typ = v.get("type")
            conf = v.get("confidence")
            out.append(
                {
                    "year": year.strip(),
                    "type": typ if isinstance(typ, str) and typ.strip() else "other",
                    "confidence": float(conf) if isinstance(conf, (int, float)) else None,
                }
            )
    return out
