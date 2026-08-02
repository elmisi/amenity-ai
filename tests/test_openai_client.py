from __future__ import annotations

import base64

from archiver import openai_client
from archiver.openai_client import MAX_TOKENS, OpenAICompatBackend, mime_from_b64
from archiver.providers import provider_by_name

PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)


def _capture(monkeypatch, response):
    captured = {}

    def fake_post(url, payload, *, timeout_s):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout_s"] = timeout_s
        return response

    monkeypatch.setattr(openai_client, "_post_json", fake_post)
    return captured


def _ok(text="hello"):
    return {
        "model": "m",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
    }


def test_mime_is_sniffed_from_magic_bytes():
    assert mime_from_b64(PNG_1X1) == "image/png"
    jpeg = base64.b64encode(b"\xff\xd8\xff\xe0somejunk").decode("ascii")
    assert mime_from_b64(jpeg) == "image/jpeg"
    assert mime_from_b64("!!!not base64!!!") == "image/png"


def test_images_are_sent_as_openai_image_url_parts(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    backend = OpenAICompatBackend("http://vllm.invalid:8000", provider_by_name("vllm"))

    result = backend.generate(prompt="describe", model="qwen3.6-27b", images_b64=[PNG_1X1])

    assert result.success
    content = captured["payload"]["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_text_only_prompt_stays_a_plain_string(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    backend = OpenAICompatBackend("http://vllm.invalid:8000", provider_by_name("vllm"))

    backend.generate(prompt="hi", model="m")

    assert captured["payload"]["messages"][0]["content"] == "hi"


def test_each_provider_switches_reasoning_off_with_its_own_lever(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    OpenAICompatBackend("http://ds4.invalid", provider_by_name("ds4")).generate(
        prompt="hi", model="deepseek-v4-flash", think=False
    )
    assert captured["payload"]["reasoning_effort"] == "low"

    # vLLM accepts reasoning_effort and ignores it: it needs its own lever.
    captured = _capture(monkeypatch, _ok())
    OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm")).generate(
        prompt="hi", model="qwen3.6-27b", think=False
    )
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in captured["payload"]


def test_reasoning_stays_on_when_think_was_not_requested(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm")).generate(
        prompt="hi", model="qwen3.6-27b"
    )
    assert "chat_template_kwargs" not in captured["payload"]


def test_max_tokens_is_capped_by_declared_context_length(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))

    backend.generate(prompt="hi", model="m", max_model_len=2048)

    assert captured["payload"]["max_tokens"] == 2048

    captured = _capture(monkeypatch, _ok())
    backend.generate(prompt="hi", model="m", max_model_len=131072)
    assert captured["payload"]["max_tokens"] == MAX_TOKENS


def test_ollama_only_arguments_never_reach_the_wire(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    backend = OpenAICompatBackend("http://vllm.invalid:8000/", provider_by_name("vllm"))

    backend.generate(
        prompt="hi",
        model="m",
        timeout_s=180.0,
        response_format="json",
        keep_alive="5m",
        options={"temperature": 0, "num_predict": 400},
    )

    payload = captured["payload"]
    assert captured["url"] == "http://vllm.invalid:8000/v1/chat/completions"
    assert captured["timeout_s"] == 180.0
    assert payload["temperature"] == 0
    assert "response_format" not in payload
    assert "keep_alive" not in payload
    assert "num_predict" not in payload


def test_truncation_by_length_is_reported_as_an_error(monkeypatch):
    _capture(monkeypatch, {
        "choices": [{"message": {"content": "part"}, "finish_reason": "length"}]
    })
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "truncated" in (result.error or "")


def test_reasoning_field_is_never_read_as_content(monkeypatch):
    # Verificato sul campo: qwen3.6-27b riempie "reasoning" lasciando
    # "content" a null finché non ha finito di ragionare.
    _capture(monkeypatch, {
        "choices": [{
            "message": {"content": None, "reasoning": "The user has provided an image"},
            "finish_reason": "stop",
        }]
    })
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "empty content" in (result.error or "")


def test_server_error_payload_becomes_an_error_response(monkeypatch):
    _capture(monkeypatch, {"error": {"message": "model not found"}})
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "model not found" in (result.error or "")


def test_empty_choices_is_reported_as_malformed(monkeypatch):
    _capture(monkeypatch, {"choices": []})
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "malformed" in (result.error or "")


def test_non_dict_json_body_is_reported_as_malformed(monkeypatch):
    _capture(monkeypatch, [])
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "malformed" in (result.error or "")


def test_transport_exception_becomes_an_error_response(monkeypatch):
    def boom(url, payload, *, timeout_s):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(openai_client, "_post_json", boom)
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "ConnectionRefusedError" in (result.error or "")
