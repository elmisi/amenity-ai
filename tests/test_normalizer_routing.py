from __future__ import annotations

from pathlib import Path

from archiver import normalizer
from archiver.ollama_client import OllamaGenerateResult
from archiver.scanner import ScanItem
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)


def _item() -> ScanItem:
    return ScanItem(
        path=Path("/tmp/doc.pdf"),
        kind="pdf",
        size_bytes=10,
        mtime_iso="2026-01-01T00:00:00",
        status="scanned",
        summary_long="A receipt for electrical work.",
        facts_json="{}",
    )


def test_normalize_items_threads_ds4_base_url(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(
            response='[{"path": "doc_1", "category": "unknown", "reference_year": null, '
                     '"proposed_name": "doc", "summary": "s", "confidence": 0.9}]',
            model=kwargs["model"],
            done=True,
        )

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    res = normalizer.normalize_items(
        items=[_item()],
        model="ds4:deepseek-v4-flash",
        base_url="http://localhost:11434",
        taxonomy=_TAXONOMY,
        output_language="en",
        filename_separator="space",
        chunk_size=1,
        ds4_base_url="http://localhost:8000",
    )
    assert captured["ds4_base_url"] == "http://localhost:8000"
    assert captured["model"] == "ds4:deepseek-v4-flash"
    assert res.error is None


def test_normalize_items_default_ds4_empty(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response="[]", model=kwargs["model"], done=True)

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    normalizer.normalize_items(
        items=[_item()],
        model="gemma3:1b",
        base_url="http://localhost:11434",
        taxonomy=_TAXONOMY,
        output_language="en",
        filename_separator="space",
        chunk_size=1,
    )
    assert captured["ds4_base_url"] == ""
