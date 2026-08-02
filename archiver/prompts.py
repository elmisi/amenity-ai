"""LLM prompt templates for document analysis and classification.

This module centralizes all LLM prompts to make them easier to maintain,
test, and potentially customize.
"""
from __future__ import annotations

from typing import Any


def build_json_repair_prompt(*, snippet: str) -> str:
    """Build a prompt to repair malformed JSON output from an LLM."""
    return f"""
You must output VALID JSON only (no code fences, no extra text).
Fix the following output into a single JSON object. Keep the same keys and values as much as possible.

Broken output:
\"\"\"{snippet}\"\"\"
"""


def _build_language_line_classify(output_language: str) -> str:
    """Build the language instruction line for classification prompts."""
    if output_language == "it":
        return "Output language: Italian"
    if output_language == "en":
        return "Output language: English"
    return "Output language: match the input document language (if unclear: English)"


def build_classify_prompt(
    *,
    categories: list[str],
    taxonomy_block: str,
    filename: str,
    mtime_iso: str,
    reference_year_hint: str | None,
    category_hint: str | None,
    content: str,
    output_language: str,
) -> str:
    """Build a prompt for document classification (legacy single-file mode)."""
    year_hint_line = f"reference_year_hint: {reference_year_hint}" if reference_year_hint else "reference_year_hint: null"
    category_hint_line = f"category_hint: {category_hint}" if category_hint else "category_hint: null"
    language_line = _build_language_line_classify(output_language)

    return f"""
You are a document archiving assistant. Reply with VALID JSON only (no extra text).

Goal:
- understand what the document is about
- choose a category from: {categories}
- taxonomy (meaning + examples):
{taxonomy_block}
- if category_hint is present, you may use it unless the content clearly indicates a different category
- estimate the reference year (reference_year) the document refers to
- estimate the production year (production_year) (if unknown: null)
- extract structured facts for later normalization (language, doc_type, tags, people, organizations, date candidates)
- write a richer but compact summary_long (3-6 sentences)
- propose a meaningful, descriptive file name (proposed_name) using words separated by spaces (not underscores)
  - use 6-12 words when possible (not too short)
  - include key entities (company/person), and month/period if present
  - copy proper names as-is (do not guess spellings; if uncertain, omit the entity)
  - do NOT include category/year unless there is no other useful info
  - do NOT include generic words like "this document", "text", "image"
  - keep it readable (avoid random IDs)
- {language_line}
- if unsure, set low confidence and provide skip_reason
- if reference_year_hint is present, use it ONLY if the content doesn't clearly contradict it

Input:
filename: {filename}
mtime_iso: {mtime_iso}
{year_hint_line}
{category_hint_line}
content:
\"\"\"{content}\"\"\"

Output JSON schema:
{{
  "language": "it"|"en"|"unknown",
  "doc_type": string,
  "tags": string[],
  "people": string[],
  "organizations": string[],
  "date_candidates": [{{"year": string, "type": "reference"|"production"|"other", "confidence": number}}],
  "summary_long": string,
  "summary": string,
  "category": string,
  "reference_year": string|null,
  "production_year": string|null,
  "proposed_name": string,
  "confidence": number,
  "skip_reason": string|null
}}
"""


def _build_language_line_facts(output_language: str) -> str:
    """Build the language instruction line for facts extraction prompts."""
    if output_language == "it":
        return (
            "Output language: Italian. All generated text fields MUST be Italian "
            "(purpose, doc_type, tags, summary_long, skip_reason). "
            "Keep proper names (people/orgs/addresses/identifiers) as-is."
        )
    if output_language == "en":
        return (
            "Output language: English. All generated text fields MUST be English "
            "(purpose, doc_type, tags, summary_long, skip_reason). "
            "Keep proper names (people/orgs/addresses/identifiers) as-is."
        )
    return "Output language: match the input document language (if unclear: English)"


def build_facts_extraction_prompt(
    *,
    filename: str,
    mtime_iso: str,
    year_hint_filename: str | None,
    year_hint_text: str | None,
    content: str,
    output_language: str,
) -> str:
    """Build a prompt for extracting facts from a document (phase 1)."""
    language_line = _build_language_line_facts(output_language)

    return f"""
You are a document understanding assistant. Reply with VALID JSON only (no extra text).

Goal:
- Extract key facts from the document content below.
- Do NOT classify or propose a filename in this step.
- Prefer precision over brevity: if a value is present, copy it exactly; do not guess.
- Keep generated text compact; avoid verbose explanations.
- Content starting with IMAGE_CAPTION: is a picture, not a paper document. Describe WHAT IT
  DEPICTS and set purpose to the subject itself (e.g. "celtic triskele symbol", "mountain
  landscape"). Having no invoice-like fields is NORMAL for a picture: never set skip_reason
  for that reason alone.
- Use skip_reason ONLY when there is no readable content at all.
- {language_line}

Inputs:
filename: {filename}
mtime_iso: {mtime_iso}
year_hint_filename: {year_hint_filename or "null"}
year_hint_text: {year_hint_text or "null"}
content:
\"\"\"{content}\"\"\"

Output JSON schema:
{{
  "language": "it"|"en"|"unknown",
  "doc_type": string|null,
  "purpose": string,        // WHAT this document IS (e.g. "electricity bill", "employment contract", "ID card photo"). NOT what you are doing with it. Do NOT mention extraction/classification/renaming.
  "tags": string[],
  "people": string[],
  "organizations": string[],
  "addresses": string[],
  "amounts": [{{"value": number, "currency": string, "raw": string}}],
  "identifiers": [{{"type": string, "value": string}}],
  "date_candidates": [{{"year": string, "type": "reference"|"production"|"other", "confidence": number, "source": "filename"|"content"}}],
  "summary_long": string,   // 2-4 sentences, include only the highest-signal values (who/what/when/how much/ids)
  "confidence": number,
  "skip_reason": string|null
}}
"""


def _build_language_line_normalize(output_language: str) -> str:
    """Build the language instruction line for normalization prompts."""
    if output_language == "it":
        return (
            "Output language: Italian. All generated text fields MUST be Italian "
            "(category labels come from taxonomy; summary/proposed_name should be Italian). "
            "Keep proper names as-is."
        )
    if output_language == "en":
        return (
            "Output language: English. All generated text fields MUST be English "
            "(category labels come from taxonomy; summary/proposed_name should be English). "
            "Keep proper names as-is."
        )
    return "Output language: match each document language; if unclear: English"


def build_normalize_batch_prompt(
    *,
    allowed_categories: list[str],
    taxonomy_block: str,
    separator_description: str,
    payload_json: str,
    output_language: str,
) -> str:
    """Build a prompt for batch document normalization (phase 2)."""
    language_line = _build_language_line_normalize(output_language)

    return f"""
You are a document archiving assistant. Reply with VALID JSON only.

Task:
- Given a batch of documents described by extracted facts (not the raw file content),
  classify and rename with maximum output quality.
- You MAY change category and reference_year if a better choice is supported by the facts.
- Produce consistent, uniform naming across the batch by using coherent templates per document cluster.
  Example: similar utility bills should share the same naming pattern (same ordering, same fields).

Constraints:
- category MUST be one of: {allowed_categories}
- proposed_name MUST be descriptive, 6-14 words when possible.
- Include key entities (organization/person) and a date/period if available in the facts or summary.
- Copy proper names as-is; do NOT guess spellings. If uncertain, omit the entity.
- Use {separator_description} between words (no mixed separators). Do NOT put separators inside a word.
- Do NOT include generic words like "document", "file", "text", "image".
- Do NOT include category/year in the name unless there is no other useful info.
- {language_line}

Taxonomy:
{taxonomy_block}

Input (JSON list):
{payload_json}

Output JSON schema (JSON list, same length as input, preserve 'path'):
[
  {{
    "path": string,
    "category": string,
    "reference_year": string|null,
    "proposed_name": string,
    "summary": string,
    "confidence": number|null
  }}
]
"""
