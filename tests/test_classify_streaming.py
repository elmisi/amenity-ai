"""Per-chunk progress callbacks for classification.

A mass classification of ~1000 files is ~90 requests over ~20 minutes; until
these callbacks existed, the UI marked every row "classifying" at minute zero
and applied every result at the very end, so the run was silent for its whole
duration and a crash lost everything.
"""
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
            path=Path(f"/tmp/f{i}.pdf"), kind="pdf", size_bytes=1,
            mtime_iso="2026-01-01T00:00:00", status="scanned",
            summary_long=f"documento {i}", facts_json="{}",
        )
        for i in range(n)
    ]


def _rows(count: int) -> str:
    return json.dumps(
        [
            {"path": f"doc_{i + 1}", "category": "personal", "reference_year": "2024",
             "proposed_name": f"doc {i}", "summary": "s"}
            for i in range(count)
        ]
    )


def _run(items, *, chunk_size=2, on_chunk_start=None, on_chunk_done=None,
         should_cancel=None, limiter=None):
    return normalizer.normalize_items(
        items=items, model="vllm:m", provider_urls=URLS, taxonomy=TAXONOMY,
        output_language="it", filename_separator="space", chunk_size=chunk_size,
        on_chunk_start=on_chunk_start, on_chunk_done=on_chunk_done,
        should_cancel=should_cancel, limiter=limiter,
    )


def test_each_chunk_announces_itself_and_reports_back(monkeypatch):
    monkeypatch.setattr(
        normalizer, "generate",
        lambda **kw: OllamaGenerateResult(response=_rows(2), model="m", done=True),
    )
    started: list[list[str]] = []
    done: list[tuple[list[str], dict]] = []
    lock = threading.Lock()

    result = _run(
        _items(6),
        on_chunk_start=lambda paths: (lock.acquire(), started.append(paths), lock.release()),
        on_chunk_done=lambda paths, partial: (lock.acquire(), done.append((paths, partial)), lock.release()),
    )
    assert len(started) == 3 and len(done) == 3
    assert sorted(p for chunk in started for p in chunk) == sorted(str(it.path) for it in _items(6))
    # every done callback carries its chunk's own results
    for paths, partial in done:
        assert set(partial) <= set(paths)
        assert partial, "a healthy chunk reports its rows"
    assert len(result.by_path) == 6


def test_a_failed_chunk_still_reports_done_with_what_it_has(monkeypatch):
    calls = {"n": 0}
    lock = threading.Lock()

    def fake(**kw):
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n == 1:
            return OllamaGenerateResult(response="", model="m", done=False, error="boom")
        return OllamaGenerateResult(response=_rows(1), model="m", done=True)

    monkeypatch.setattr(normalizer, "generate", fake)
    done: list[tuple[list[str], dict]] = []
    _run(
        _items(2), chunk_size=2,
        on_chunk_done=lambda paths, partial: done.append((paths, partial)),
    )
    assert len(done) == 1, "one chunk, one report, even after the single-item fallback"


def test_cancellation_stops_new_chunks_before_they_announce(monkeypatch):
    monkeypatch.setattr(
        normalizer, "generate",
        lambda **kw: OllamaGenerateResult(response=_rows(2), model="m", done=True),
    )
    started: list[list[str]] = []
    result = _run(
        _items(6),
        on_chunk_start=lambda paths: started.append(paths),
        should_cancel=lambda: True,
    )
    assert started == [], "a cancelled run must not claim rows are in flight"
    assert result.error == "Cancelled"


def test_callbacks_flow_through_the_model_fallback(monkeypatch):
    monkeypatch.setattr(
        normalizer, "generate",
        lambda **kw: OllamaGenerateResult(response=_rows(2), model="m", done=True),
    )
    done: list[tuple[list[str], dict]] = []
    result = normalizer.normalize_items_with_fallback(
        items=_items(2), models=("vllm:m",), provider_urls=URLS, taxonomy=TAXONOMY,
        output_language="it", filename_separator="space", chunk_size=2,
        on_chunk_done=lambda paths, partial: done.append((paths, partial)),
    )
    assert len(done) == 1 and result.error is None


def test_streaming_and_parallelism_compose(monkeypatch):
    gate = threading.Barrier(2, timeout=5)

    def fake(**kw):
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
        return OllamaGenerateResult(response=_rows(2), model="m", done=True)

    monkeypatch.setattr(normalizer, "generate", fake)
    done: list[int] = []
    lock = threading.Lock()
    _run(
        _items(4), limiter=ConcurrencyLimiter.from_limits({"vllm": 2}),
        on_chunk_done=lambda paths, partial: (lock.acquire(), done.append(len(partial)), lock.release()),
    )
    assert done == [2, 2]
