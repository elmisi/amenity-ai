from __future__ import annotations

from dataclasses import replace

from .extractors.registry import EXTRACTABLE_TEXT_KINDS
from .scanner import ScanItem


def can_requeue(item: ScanItem) -> bool:
    """True when requeuing this row could change its outcome.

    A skipped or error row is worth another try only if some extractor can
    handle its kind. The scanner lists every file in the folder and marks the
    zips, webp and friends skipped at listing time; requeuing those just runs
    them through the analyzer to be re-skipped, polluting the cache with
    entries that were never meant to be there.
    """
    if item.status not in {"skipped", "error"}:
        return False
    return item.kind == "image" or item.kind in EXTRACTABLE_TEXT_KINDS


def reset_item_to_pending(item: ScanItem) -> ScanItem:
    """Reset an item to `pending` and clear all scan/classify derived fields."""
    return replace(
        item,
        status="pending",
        reason=None,
        category=None,
        reference_year=None,
        proposed_name=None,
        summary=None,
        summary_long=None,
        facts_json=None,
        llm_raw_output=None,
        confidence=None,
        analysis_time_s=None,
        model_used=None,
        extract_method=None,
        extract_time_s=None,
        llm_time_s=None,
        ocr_time_s=None,
        ocr_mode=None,
        facts_time_s=None,
        facts_llm_time_s=None,
        facts_model_used=None,
        classify_time_s=None,
        classify_llm_time_s=None,
        classify_model_used=None,
    )


def mark_item_scanning(item: ScanItem) -> ScanItem:
    """Mark a pending item as being scanned and clear classification fields."""
    return replace(
        item,
        status="scanning",
        reason=None,
        category=None,
        reference_year=None,
        proposed_name=None,
        summary=None,
        llm_raw_output=None,
        classify_time_s=None,
        classify_llm_time_s=None,
        classify_model_used=None,
    )


def mark_item_classifying(item: ScanItem) -> ScanItem:
    """Mark a scanned item as being classified."""
    return replace(item, status="classifying", reason=None)


def unclassify_item(item: ScanItem) -> ScanItem:
    """Bring a classified item back to scanned, keeping facts and clearing classification."""
    return replace(
        item,
        status="scanned",
        reason=None,
        category=None,
        reference_year=None,
        proposed_name=None,
        confidence=None,
        analysis_time_s=None,
        model_used=item.facts_model_used or item.model_used,
        classify_time_s=None,
        classify_llm_time_s=None,
        classify_model_used=None,
    )

