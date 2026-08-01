from __future__ import annotations

from archiver import llm_router
from archiver.llm_backend import LLMResponse
from archiver.ollama_client import OllamaGenerateResult


def test_prefix_constant_and_predicate():
    assert llm_router.DS4_PREFIX == "ds4:"
    assert llm_router.is_ds4_model("ds4:deepseek-v4-flash")
    assert not llm_router.is_ds4_model("gemma3:1b")


def test_ds4_model_routes_to_ds4_backend(monkeypatch):
    captured = {}

    class FakeBackend:
        def __init__(self, base_url):
            captured["base_url"] = base_url

        def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return LLMResponse(text="hi", model="deepseek-v4-flash")

    monkeypatch.setattr(llm_router, "Ds4Backend", FakeBackend)

    def boom(**kwargs):
        raise AssertionError("ollama must not be called for ds4 models")

    monkeypatch.setattr(llm_router, "_ollama_generate", boom)

    res = llm_router.generate(
        model="ds4:deepseek-v4-flash",
        prompt="q",
        ds4_base_url="http://localhost:8000",
        think=False,
        options={"temperature": 0},
    )
    assert isinstance(res, OllamaGenerateResult)
    assert res.error is None
    assert res.response == "hi"
    assert res.model == "ds4:deepseek-v4-flash"  # keeps the prefixed id for cache/UI
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["kwargs"]["model"] == "deepseek-v4-flash"  # prefix stripped


def test_plain_model_routes_to_ollama(monkeypatch):
    captured = {}

    def fake_ollama(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response="ok", model="gemma3:1b", done=True)

    monkeypatch.setattr(llm_router, "_ollama_generate", fake_ollama)
    res = llm_router.generate(
        model="gemma3:1b",
        prompt="q",
        base_url="http://localhost:11434",
        ds4_base_url="http://localhost:8000",
    )
    assert res.response == "ok"
    assert captured["model"] == "gemma3:1b"
    assert "ds4_base_url" not in captured  # ollama API unchanged


def test_ds4_without_endpoint_is_error(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("no backend must be called")

    monkeypatch.setattr(llm_router, "_ollama_generate", boom)
    res = llm_router.generate(model="ds4:deepseek-v4-flash", prompt="q", ds4_base_url="")
    assert res.error
    assert "not configured" in res.error


def test_image_file_on_ds4_is_error(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    res = llm_router.generate_with_image_file(
        model="ds4:deepseek-v4-flash",
        prompt="q",
        image_path=str(img),
        ds4_base_url="http://localhost:8000",
    )
    assert res.error
    assert "vision" in res.error


def test_image_file_on_ollama_delegates(monkeypatch, tmp_path):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response="desc", model="moondream:latest", done=True)

    monkeypatch.setattr(llm_router, "_ollama_generate_with_image_file", fake)
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    res = llm_router.generate_with_image_file(
        model="moondream:latest", prompt="q", image_path=str(img), ds4_base_url="http://localhost:8000"
    )
    assert res.response == "desc"
    assert captured["model"] == "moondream:latest"
