from __future__ import annotations

import json

from archiver import analyzer
from archiver.analyzer import AnalysisConfig, _classify_from_text, _extract_facts_from_text
from archiver.ollama_client import OllamaGenerateResult
from archiver.scanner import ScanItem
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)


def test_analysis_config_has_ds4_base_url_default_empty():
    assert AnalysisConfig().ds4_base_url == ""


def _fake_generate(captured, payload_text):
    def fake(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response=payload_text, model=kwargs["model"], done=True)

    return fake


def test_classify_passes_ds4_base_url_to_generate(monkeypatch):
    captured = {}
    out = json.dumps({"category": "unknown", "reference_year": None, "proposed_name": "doc"})
    monkeypatch.setattr(analyzer, "generate", _fake_generate(captured, out))

    _classify_from_text(
        model="ds4:deepseek-v4-flash",
        content="some text",
        filename="a.pdf",
        mtime_iso="2026-01-01T00:00:00",
        base_url="http://localhost:11434",
        ds4_base_url="http://localhost:8000",
        reference_year_hint=None,
        category_hint=None,
        output_language="en",
        taxonomy=_TAXONOMY,
        filename_separator="space",
    )
    assert captured["ds4_base_url"] == "http://localhost:8000"
    assert captured["model"] == "ds4:deepseek-v4-flash"


def test_facts_passes_ds4_base_url_to_generate(monkeypatch):
    captured = {}
    out = json.dumps({"summary_long": "A letter about something.", "confidence": 0.9})
    monkeypatch.setattr(analyzer, "generate", _fake_generate(captured, out))

    res = _extract_facts_from_text(
        model="ds4:deepseek-v4-flash",
        content="some text",
        filename="a.pdf",
        mtime_iso="2026-01-01T00:00:00",
        base_url="http://localhost:11434",
        ds4_base_url="http://localhost:8000",
        year_hint_filename=None,
        year_hint_text=None,
        output_language="en",
    )
    assert captured["ds4_base_url"] == "http://localhost:8000"
    assert res.status == "scanned"


def test_extract_facts_item_falls_back_to_next_model_on_error(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_generate(**kwargs):
        model = kwargs["model"]
        calls.append(model)
        if model == "ds4:deepseek-v4-flash":
            return OllamaGenerateResult(response="", model=model, done=True, error="connection refused")
        out = json.dumps({"summary_long": "A short letter about a utility bill.", "confidence": 0.9})
        return OllamaGenerateResult(response=out, model=model, done=True)

    monkeypatch.setattr(analyzer, "generate", fake_generate)

    p = tmp_path / "note.txt"
    p.write_text("Hello, this is a plain text document used for a routing test.")

    item = ScanItem(
        path=p,
        kind="txt",
        size_bytes=p.stat().st_size,
        mtime_iso="2026-01-01T00:00:00",
        status="pending",
    )
    cfg = AnalysisConfig(text_models=("ds4:deepseek-v4-flash", "gemma3:1b"))

    res = analyzer.extract_facts_item(item, config=cfg)

    assert res.status == "scanned"
    assert res.model_used == "gemma3:1b"
    assert calls == ["ds4:deepseek-v4-flash", "gemma3:1b"]
