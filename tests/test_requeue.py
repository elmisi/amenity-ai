"""What `k` may and may not requeue.

Found in the field: the first version requeued every skipped row, including
the zips and webp the scanner marks unsupported at listing time. Those ran
through the analyzer only to be re-skipped, and ~180 entries that were never
meant to be cached ended up in the cache.
"""
from pathlib import Path

from archiver.item_mutations import can_requeue
from archiver.scanner import ScanItem


def _item(name: str, kind: str, status: str) -> ScanItem:
    return ScanItem(
        path=Path(f"/tmp/{name}"), kind=kind, size_bytes=1,
        mtime_iso="2026-01-01T00:00:00", status=status,
    )


def test_a_skipped_pdf_is_worth_another_try():
    assert can_requeue(_item("a.pdf", "pdf", "skipped"))


def test_a_skipped_image_is_worth_another_try():
    assert can_requeue(_item("a.png", "image", "skipped"))


def test_an_error_row_with_an_extractor_is_worth_another_try():
    assert can_requeue(_item("a.csv", "csv", "error"))


def test_a_zip_is_not_requeued_because_nothing_can_extract_it():
    assert not can_requeue(_item("a.zip", "zip", "skipped"))


def test_an_extensionless_file_is_not_requeued():
    assert not can_requeue(_item("README", "unknown", "skipped"))


def test_rows_that_are_not_skipped_or_error_are_left_alone():
    assert not can_requeue(_item("a.pdf", "pdf", "scanned"))
    assert not can_requeue(_item("a.pdf", "pdf", "pending"))
    assert not can_requeue(_item("a.pdf", "pdf", "classified"))
