"""Every extension the scanner includes must reach an extractor.

The analyzer's dispatch and the extractor registry were once two hand-kept
lists; they drifted, and csv/html/yaml/gpx files spent months reported as
"Unsupported file type" with their extractors sitting unused. The dispatch is
now derived from the registry, and this test pins the whole chain:
extension -> kind -> extractor.
"""
from pathlib import Path

from archiver.extractors.registry import EXTRACTABLE_TEXT_KINDS, extract_with_meta
from archiver.filetypes import infer_kind
from archiver.settings import Settings


def _included_extensions() -> tuple[str, ...]:
    return Settings(source_root=Path("/tmp/s"), archive_root=Path("/tmp/a")).include_extensions


def test_every_included_extension_maps_to_a_kind():
    for ext in _included_extensions():
        assert infer_kind(Path(f"f.{ext}")) is not None, f".{ext} has no kind"


def test_every_included_kind_reaches_an_extractor():
    for ext in _included_extensions():
        kind = infer_kind(Path(f"f.{ext}"))
        assert kind == "image" or kind in EXTRACTABLE_TEXT_KINDS, (
            f".{ext} maps to kind {kind!r}, which no extractor handles — "
            f"it would be skipped as 'Unsupported file type'"
        )


def test_the_registry_does_not_call_a_csv_unsupported(tmp_path):
    """The concrete case found in the field: 29 csv files in one real scan."""
    sample = tmp_path / "data.csv"
    sample.write_text("date,amount\n2024-01-15,42.50\n2024-02-15,43.10\n")
    text, reason, _ = extract_with_meta(kind="csv", path=sample)
    assert reason != "Unsupported file type"
    assert text and "42.50" in text
