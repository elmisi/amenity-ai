import json
import threading
from pathlib import Path

from archiver import normalizer
from archiver.concurrency import ConcurrencyLimiter
from archiver.ollama_client import OllamaGenerateResult
from archiver.scanner import ScanItem
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)
URLS = {"vllm": "http://example.invalid"}


def _items(n: int) -> list[ScanItem]:
    return [
        ScanItem(
            path=Path(f"/tmp/f{i}.pdf"),
            kind="pdf",
            size_bytes=1,
            mtime_iso="2026-01-01T00:00:00",
            status="scanned",
            summary_long=f"documento numero {i}",
            facts_json=json.dumps({"doc_type": "fattura"}),
        )
        for i in range(n)
    ]


def _rows(paths: list[str]) -> str:
    return json.dumps(
        [
            {
                "path": f"doc_{i + 1}",
                "category": "personal",
                "reference_year": "2024",
                "proposed_name": f"documento {i}",
                "summary": "riassunto",
            }
            for i, _ in enumerate(paths)
        ]
    )


def _run(items, *, chunk_size, limiter=None, should_cancel=None):
    return normalizer.normalize_items(
        items=items,
        model="vllm:m",
        provider_urls=URLS,
        taxonomy=TAXONOMY,
        output_language="it",
        filename_separator="space",
        chunk_size=chunk_size,
        limiter=limiter,
        should_cancel=should_cancel,
    )


def test_chunks_run_concurrently_up_to_the_provider_limit(monkeypatch):
    peak = 0
    live = 0
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=5)

    def fake_generate(**kwargs):
        nonlocal peak, live
        with lock:
            live += 1
            peak = max(peak, live)
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
        with lock:
            live -= 1
        return OllamaGenerateResult(response=_rows(["a", "b"]), model="m", done=True)

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    _run(_items(4), chunk_size=2, limiter=ConcurrencyLimiter.from_limits({"vllm": 2}))
    assert peak == 2


def test_without_a_limiter_chunks_stay_sequential(monkeypatch):
    peak = 0
    live = 0
    lock = threading.Lock()

    def fake_generate(**kwargs):
        nonlocal peak, live
        with lock:
            live += 1
            peak = max(peak, live)
        with lock:
            live -= 1
        return OllamaGenerateResult(response=_rows(["a", "b"]), model="m", done=True)

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    _run(_items(4), chunk_size=2)
    assert peak == 1


def test_one_failing_chunk_does_not_abandon_the_others(monkeypatch):
    """The old loop returned on the first error, so every later chunk was lost."""
    calls = {"n": 0}
    lock = threading.Lock()

    def fake_generate(**kwargs):
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n == 1:
            return OllamaGenerateResult(response="", model="m", done=False, error="boom")
        return OllamaGenerateResult(response=_rows(["a", "b"]), model="m", done=True)

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    result = _run(_items(6), chunk_size=2)
    assert result.by_path, "the healthy chunks must still produce output"


def test_a_cancelled_run_reports_cancelled_over_any_other_error(monkeypatch):
    def fake_generate(**kwargs):
        return OllamaGenerateResult(response="", model="m", done=False, error="boom")

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    result = _run(_items(4), chunk_size=2, should_cancel=lambda: True)
    assert result.error == "Cancelled"


def test_results_from_every_chunk_are_merged(monkeypatch):
    def fake_generate(**kwargs):
        return OllamaGenerateResult(response=_rows(["a", "b"]), model="m", done=True)

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    result = _run(
        _items(6), chunk_size=2, limiter=ConcurrencyLimiter.from_limits({"vllm": 3})
    )
    assert len(result.by_path) == 6
    assert result.error is None
