"""Vision must go through the router, not talk to Ollama directly.

Before the redesign `extractors/image.py` called ollama_client, so a vision
model served by vLLM would have reached Ollama with its prefix attached.
The doctor would call it available and then it would fail.
"""
from __future__ import annotations

from pathlib import Path

from archiver.extractors import image as image_extractor
from archiver.ollama_client import OllamaGenerateResult

URLS = {"ollama": "http://ollama.invalid:11434", "vllm": "http://vllm.invalid:8000", "ds4": ""}


def _png(tmp_path: Path) -> Path:
    path = tmp_path / "a.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    return path


def test_caption_routes_a_vllm_model_through_the_router(monkeypatch, tmp_path):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response="a cat", model=kwargs["model"], done=True)

    monkeypatch.setattr(image_extractor, "generate_with_image_file", fake)

    caption, meta = image_extractor.caption_image(
        _png(tmp_path),
        vision_models=("vllm:qwen3.6-27b",),
        prompt="describe",
        provider_urls=URLS,
    )

    assert caption == "a cat"
    assert meta.vision_model_used == "vllm:qwen3.6-27b"
    assert captured["provider_urls"] == URLS
    # The prefix stays on: resolving it is the router's job.
    assert captured["model"] == "vllm:qwen3.6-27b"


def test_caption_falls_back_to_the_next_vision_model(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "vllm:qwen3.6-27b":
            return OllamaGenerateResult(response="", model=kwargs["model"], done=False,
                                        error="vllm: endpoint not configured")
        return OllamaGenerateResult(response="a dog", model=kwargs["model"], done=True)

    monkeypatch.setattr(image_extractor, "generate_with_image_file", fake)

    caption, _ = image_extractor.caption_image(
        _png(tmp_path),
        vision_models=("vllm:qwen3.6-27b", "ollama:moondream:latest"),
        prompt="describe",
        provider_urls=URLS,
    )

    assert caption == "a dog"
    assert calls == ["vllm:qwen3.6-27b", "ollama:moondream:latest"]
