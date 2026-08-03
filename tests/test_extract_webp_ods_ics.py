"""webp, ods and ics — three types whose plumbing already half-existed.

webp only had to be let into the image pipeline: the vision path already
sniffs its MIME type from the magic bytes. ods shares content.xml with odt,
which the office extractor already parses. ics is plain text whose dates,
reformatted with dashes, feed the analyzer's year hint directly.
"""
from __future__ import annotations

import base64
import zipfile
from pathlib import Path

from archiver.extractors.registry import extract_with_meta
from archiver.extractors.textish_ics import extract_ics_text
from archiver.filetypes import infer_kind
from archiver.openai_client import mime_from_b64


# --- webp ---------------------------------------------------------------

def test_webp_is_an_image_kind():
    assert infer_kind(Path("photo.webp")) == "image"


def test_webp_mime_is_already_sniffed_for_the_vision_path():
    header = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    assert mime_from_b64(base64.b64encode(header).decode()) == "image/webp"


# --- ods ----------------------------------------------------------------

_ODS_CONTENT = (
    '<office:document-content'
    ' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
    ' xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
    ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
    "<office:body><office:spreadsheet>"
    '<table:table table:name="Spese 2024">'
    "<table:table-row>"
    "<table:table-cell><text:p>data</text:p></table:table-cell>"
    "<table:table-cell><text:p>importo</text:p></table:table-cell>"
    "</table:table-row>"
    "<table:table-row>"
    "<table:table-cell><text:p>2024-03-15</text:p></table:table-cell>"
    "<table:table-cell><text:p>120,50</text:p></table:table-cell>"
    "</table:table-row>"
    "</table:table></office:spreadsheet></office:body>"
    "</office:document-content>"
)


def _make_ods(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        zf.writestr("content.xml", _ODS_CONTENT)
    return path


def test_ods_cells_are_extracted(tmp_path):
    p = _make_ods(tmp_path / "spese.ods")
    text, method, meta = extract_with_meta(kind="ods", path=p)
    assert method == "ods" and meta is not None
    assert "importo" in text and "120,50" in text and "2024-03-15" in text


def test_a_corrupt_ods_reports_no_text(tmp_path):
    p = tmp_path / "broken.ods"
    p.write_bytes(b"PK\x03\x04not a real zip")
    text, _, _ = extract_with_meta(kind="ods", path=p)
    assert text is None


# --- ics ----------------------------------------------------------------

ICS = "\r\n".join(
    [
        "BEGIN:VCALENDAR",
        "X-WR-CALNAME:Condominio",
        "BEGIN:VEVENT",
        # Folded mid-word, as real producers do (RFC 5545 3.1: the fold is
        # CRLF + one space, and that space is a marker, not content).
        "SUMMARY:Assemblea condominiale straordinaria con approvazione del bilan",
        " cio consuntivo",
        "DTSTART:20240712T180000Z",
        "LOCATION:Sala riunioni\\, via Roma 5",
        "DESCRIPTION:Ordine del giorno: lavori facciata",
        "END:VEVENT",
        "BEGIN:VEVENT",
        "SUMMARY:Scadenza rata condominio",
        "DTSTART;VALUE=DATE:20241001",
        "END:VEVENT",
        "END:VCALENDAR",
        "",
    ]
)


def test_ics_extracts_events_with_readable_dates(tmp_path):
    p = tmp_path / "cal.ics"
    p.write_text(ICS, encoding="utf-8")
    text = extract_ics_text(p, max_chars=15000)
    assert "Condominio" in text
    # dashes matter: "20240712" defeats the year-hint regex, "2024-07-12" feeds it
    assert "2024-07-12" in text and "2024-10-01" in text
    assert "20240712" not in text


def test_ics_unfolds_wrapped_lines_and_unescapes(tmp_path):
    p = tmp_path / "cal.ics"
    p.write_text(ICS, encoding="utf-8")
    text = extract_ics_text(p, max_chars=15000)
    assert "approvazione del bilancio consuntivo" in text
    assert "Sala riunioni, via Roma 5" in text
    assert "\\," not in text


def test_ics_with_no_events_still_returns_nothing_useful(tmp_path):
    p = tmp_path / "cal.ics"
    p.write_text("BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n", encoding="utf-8")
    assert extract_ics_text(p, max_chars=1000) is None


def test_an_empty_ics_returns_none(tmp_path):
    p = tmp_path / "cal.ics"
    p.write_bytes(b"")
    assert extract_ics_text(p, max_chars=1000) is None


def test_the_registry_routes_ics(tmp_path):
    p = tmp_path / "cal.ics"
    p.write_text(ICS, encoding="utf-8")
    text, method, meta = extract_with_meta(kind="ics", path=p)
    assert method == "ics" and meta is not None and "Assemblea" in text
