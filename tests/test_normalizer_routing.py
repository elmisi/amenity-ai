from __future__ import annotations

from pathlib import Path

from archiver import normalizer
from archiver.ollama_client import OllamaGenerateResult
from archiver.scanner import ScanItem
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)

URLS = {"ollama": "http://ollama.invalid:11434", "vllm": "http://vllm.invalid:8000", "ds4": ""}


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


def test_normalize_items_threads_provider_urls(monkeypatch):
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
        model="vllm:qwen3.6-27b",
        provider_urls=URLS,
        taxonomy=_TAXONOMY,
        output_language="en",
        filename_separator="space",
        chunk_size=1,
    )
    assert captured["provider_urls"] == URLS
    assert captured["model"] == "vllm:qwen3.6-27b"
    assert res.error is None


def test_normalize_items_defaults_to_no_providers(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response="[]", model=kwargs["model"], done=True)

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    normalizer.normalize_items(
        items=[_item()],
        model="ollama:gemma3:1b",
        taxonomy=_TAXONOMY,
        output_language="en",
        filename_separator="space",
        chunk_size=1,
    )
    assert captured["provider_urls"] == {}


def test_normalize_items_with_fallback_tries_next_model_on_error(monkeypatch):
    calls: list[str] = []

    def fake_generate(**kwargs):
        model = kwargs["model"]
        calls.append(model)
        if model == "vllm:qwen3.6-27b":
            return OllamaGenerateResult(response="", model=model, done=True, error="connection refused")
        return OllamaGenerateResult(
            response='[{"path": "doc_1", "category": "unknown", "reference_year": null, '
                     '"proposed_name": "doc", "summary": "s", "confidence": 0.9}]',
            model=model,
            done=True,
        )

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    res = normalizer.normalize_items_with_fallback(
        items=[_item()],
        models=("vllm:qwen3.6-27b", "ollama:gemma3:1b"),
        provider_urls=URLS,
        taxonomy=_TAXONOMY,
        output_language="en",
        filename_separator="space",
        chunk_size=1,
    )
    assert res.error is None
    assert calls == ["vllm:qwen3.6-27b", "ollama:gemma3:1b"]


def test_normalize_items_with_fallback_stops_immediately_on_cancelled(monkeypatch):
    calls: list[str] = []

    def fake_normalize_items(*, model, **_kwargs):
        calls.append(model)
        return normalizer.NormalizationResult(by_path={}, model_used=model, error="Cancelled")

    monkeypatch.setattr(normalizer, "normalize_items", fake_normalize_items)
    res = normalizer.normalize_items_with_fallback(
        items=[_item()],
        models=("ollama:gemma3:1b", "vllm:qwen3.6-27b"),
        provider_urls=URLS,
        taxonomy=_TAXONOMY,
        output_language="en",
        filename_separator="space",
        chunk_size=1,
        should_cancel=lambda: True,
    )
    assert res.error == "Cancelled"
    assert calls == ["ollama:gemma3:1b"]
