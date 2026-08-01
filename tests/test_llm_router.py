from __future__ import annotations

import base64

from archiver import llm_router
from archiver.llm_backend import LLMResponse

URLS = {
    "ollama": "http://ollama.invalid:11434",
    "vllm": "http://vllm.invalid:8000",
    "ds4": "",
}


class _Spy:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, base_url, spec=None):
        self.calls.append({"base_url": base_url, "spec": spec})
        return self

    def generate(self, **kwargs):
        self.calls[-1].update(kwargs)
        return self.response


def test_bare_legacy_id_routes_to_ollama(monkeypatch):
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OllamaBackend", spy)

    result = llm_router.generate(model="qwen3:8b", prompt="hi", provider_urls=URLS)

    assert result.response == "ok"
    assert spy.calls[0]["base_url"] == URLS["ollama"]
    assert spy.calls[0]["model"] == "qwen3:8b"


def test_prefixed_ollama_id_strips_the_prefix_before_the_call(monkeypatch):
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OllamaBackend", spy)

    llm_router.generate(model="ollama:qwen3:8b", prompt="hi", provider_urls=URLS)

    assert spy.calls[0]["model"] == "qwen3:8b"


def test_vllm_id_routes_to_the_openai_backend_with_its_spec(monkeypatch):
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OpenAICompatBackend", spy)

    result = llm_router.generate(
        model="vllm:qwen3.6-27b", prompt="hi", provider_urls=URLS
    )

    assert result.response == "ok"
    assert spy.calls[0]["base_url"] == URLS["vllm"]
    assert spy.calls[0]["spec"].name == "vllm"
    assert spy.calls[0]["model"] == "qwen3.6-27b"


def test_unconfigured_provider_fails_explicitly_instead_of_falling_back():
    result = llm_router.generate(model="ds4:whatever", prompt="hi", provider_urls=URLS)
    assert result.done is False
    assert "ds4" in (result.error or "")
    assert "configurat" in (result.error or "")


def test_image_file_goes_to_ollama_as_base64(monkeypatch, tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OllamaBackend", spy)

    llm_router.generate_with_image_file(
        model="ollama:llava:7b", prompt="what", image_path=str(png), provider_urls=URLS
    )

    sent = spy.calls[0]["images_b64"][0]
    assert base64.b64decode(sent).startswith(b"\x89PNG")


def test_image_file_now_works_on_openai_compat_providers(monkeypatch, tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OpenAICompatBackend", spy)

    result = llm_router.generate_with_image_file(
        model="vllm:qwen3.6-27b", prompt="what", image_path=str(png), provider_urls=URLS
    )

    assert result.done is True
    assert spy.calls[0]["images_b64"]
