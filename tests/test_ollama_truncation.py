from __future__ import annotations

from archiver import ollama_client
from archiver.normalizer import _normalize_options
from archiver.ollama_client import OllamaBackend


def _response(text: str = '{"ok": true}', done_reason: str = "stop") -> dict:
    return {"model": "qwen3:8b", "response": text, "done": True, "done_reason": done_reason}


def test_generate_flags_truncated_output_as_error(monkeypatch):
    monkeypatch.setattr(
        ollama_client, "_post_json",
        lambda url, payload, *, timeout_s: _response('{"partial": "cut off mid-str', done_reason="length"),
    )
    resp = OllamaBackend("http://localhost:11434").generate(prompt="q", model="qwen3:8b")
    assert not resp.success
    assert "truncated" in (resp.error or "")


def test_generate_accepts_normal_completion(monkeypatch):
    monkeypatch.setattr(
        ollama_client, "_post_json",
        lambda url, payload, *, timeout_s: _response(),
    )
    resp = OllamaBackend("http://localhost:11434").generate(prompt="q", model="qwen3:8b")
    assert resp.success
    assert resp.text == '{"ok": true}'


def test_generate_without_done_reason_still_succeeds(monkeypatch):
    # Older Ollama versions omit done_reason entirely.
    monkeypatch.setattr(
        ollama_client, "_post_json",
        lambda url, payload, *, timeout_s: {"model": "gemma3:1b", "response": "hi", "done": True},
    )
    resp = OllamaBackend("http://localhost:11434").generate(prompt="q", model="gemma3:1b")
    assert resp.success


def test_normalize_options_scale_with_batch_size():
    single = _normalize_options(1)
    batch = _normalize_options(12)
    assert single["temperature"] == 0
    assert single["num_predict"] >= 800
    assert batch["num_predict"] >= 12 * 250       # one row per item needs its own budget
    assert batch["num_predict"] > single["num_predict"]
