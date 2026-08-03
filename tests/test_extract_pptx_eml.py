"""Extraction from .pptx and .eml — the two types a real scan surfaced.

A 1200-file scan of a Downloads folder held 6 pptx and 11 eml with no
extractor. Both formats open with the standard library alone: pptx is a zip
of XML (like kmz), eml is what the `email` module exists for.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from archiver.extractors.registry import extract_with_meta
from archiver.extractors.textish_eml import extract_eml_text
from archiver.extractors.textish_pptx import extract_pptx_text

_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _slide(*texts: str) -> str:
    runs = "".join(
        f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in texts
    )
    return (
        f'<p:sld xmlns:a="{_A}" xmlns:p="{_P}">'
        f"<p:cSld><p:spTree><p:sp><p:txBody>{runs}</p:txBody></p:sp>"
        f"</p:spTree></p:cSld></p:sld>"
    )


def _make_pptx(path: Path, slides: list[str], *, core: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for i, xml in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{i}.xml", xml)
        if core:
            zf.writestr("docProps/core.xml", core)
    return path


CORE = (
    '<cp:coreProperties'
    ' xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
    ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
    ' xmlns:dcterms="http://purl.org/dc/terms/">'
    "<dc:title>Piano commerciale 2024</dc:title>"
    "<dc:creator>Mario Bianchi</dc:creator>"
    "<dcterms:created>2024-03-15T10:00:00Z</dcterms:created>"
    "</cp:coreProperties>"
)


def test_pptx_extracts_slides_in_order(tmp_path):
    p = _make_pptx(
        tmp_path / "deck.pptx",
        [_slide("Budget preventivo"), _slide("Ricavi attesi", "Costi fissi")],
    )
    text = extract_pptx_text(p, max_chars=15000)
    assert text is not None
    assert text.index("Budget preventivo") < text.index("Ricavi attesi")
    assert "Costi fissi" in text


def test_pptx_carries_title_author_and_date(tmp_path):
    p = _make_pptx(tmp_path / "deck.pptx", [_slide("contenuto")], core=CORE)
    text = extract_pptx_text(p, max_chars=15000)
    assert "Piano commerciale 2024" in text
    assert "Mario Bianchi" in text
    assert "2024-03-15" in text  # the year hint the analyzer feeds on


def test_pptx_slide_ten_sorts_after_slide_two(tmp_path):
    slides = [_slide(f"slide numero {i}") for i in range(1, 12)]
    p = _make_pptx(tmp_path / "deck.pptx", slides)
    text = extract_pptx_text(p, max_chars=15000)
    assert text.index("slide numero 2") < text.index("slide numero 10")


def test_pptx_respects_max_chars(tmp_path):
    p = _make_pptx(tmp_path / "deck.pptx", [_slide("parola " * 500)] * 5)
    text = extract_pptx_text(p, max_chars=300)
    assert text is not None and len(text) <= 300


def test_a_corrupt_pptx_returns_none(tmp_path):
    p = tmp_path / "broken.pptx"
    p.write_bytes(b"PK\x03\x04not really a zip")
    assert extract_pptx_text(p, max_chars=1000) is None


EML = b"""\
From: Rossi Energia <fatture@example.invalid>
To: mario.bianchi@example.invalid
Subject: Fattura marzo 2024
Date: Mon, 18 Mar 2024 09:30:00 +0100
Content-Type: text/plain; charset=utf-8

Buongiorno,
in allegato la fattura del mese di marzo, importo 120,50 euro.
Cordiali saluti
"""


def test_eml_carries_headers_and_body():
    def check(text: str) -> None:
        assert "Rossi Energia" in text
        assert "Fattura marzo 2024" in text
        assert "18 Mar 2024" in text  # year hint
        assert "120,50" in text

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mail.eml"
        p.write_bytes(EML)
        text = extract_eml_text(p, max_chars=15000)
        assert text is not None
        check(text)


def test_eml_falls_back_to_stripped_html(tmp_path):
    raw = (
        b"From: a@example.invalid\r\n"
        b"Subject: Conferma ordine\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n\r\n"
        b"<html><body><p>Ordine <b>confermato</b> il 2 aprile</p></body></html>\r\n"
    )
    p = tmp_path / "mail.eml"
    p.write_bytes(raw)
    text = extract_eml_text(p, max_chars=15000)
    assert "Ordine" in text and "confermato" in text
    assert "<b>" not in text


def test_eml_lists_attachment_names(tmp_path):
    raw = (
        b"From: a@example.invalid\r\n"
        b"Subject: Documenti\r\n"
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: multipart/mixed; boundary="B"\r\n\r\n'
        b"--B\r\n"
        b"Content-Type: text/plain\r\n\r\n"
        b"vedi allegato\r\n"
        b"--B\r\n"
        b"Content-Type: application/pdf\r\n"
        b'Content-Disposition: attachment; filename="fattura_marzo.pdf"\r\n\r\n'
        b"%PDF-fake\r\n"
        b"--B--\r\n"
    )
    p = tmp_path / "mail.eml"
    p.write_bytes(raw)
    text = extract_eml_text(p, max_chars=15000)
    assert "fattura_marzo.pdf" in text


def test_an_empty_eml_returns_none(tmp_path):
    p = tmp_path / "empty.eml"
    p.write_bytes(b"")
    assert extract_eml_text(p, max_chars=1000) is None


def test_the_registry_routes_both_kinds(tmp_path):
    deck = _make_pptx(tmp_path / "deck.pptx", [_slide("bilancio annuale")])
    text, method, meta = extract_with_meta(kind="pptx", path=deck)
    assert method == "pptx" and "bilancio annuale" in text and meta is not None

    mail = tmp_path / "mail.eml"
    mail.write_bytes(EML)
    text, method, meta = extract_with_meta(kind="eml", path=mail)
    assert method == "eml" and "Fattura marzo 2024" in text and meta is not None
