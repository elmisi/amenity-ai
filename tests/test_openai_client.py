from __future__ import annotations

import pytest

from archiver import openai_client
from archiver.openai_client import Ds4Backend


def _ok_response(content: str = '{"ok": true}') -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "secret chain of thought",
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_generate_builds_openai_payload(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, timeout_s):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout_s"] = timeout_s
        return _ok_response()

    monkeypatch.setattr(openai_client, "_post_json", fake_post)
    backend = Ds4Backend("http://localhost:8000/")
    resp = backend.generate(
        prompt="hello",
        model="deepseek-v4-flash",
        timeout_s=180.0,
        response_format="json",
        think=False,
        keep_alive="5m",
        options={"temperature": 0, "num_predict": 400},
    )

    assert resp.success
    assert resp.text == '{"ok": true}'
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    p = captured["payload"]
    assert p["model"] == "deepseek-v4-flash"
    assert p["messages"] == [{"role": "user", "content": "hello"}]
    assert p["max_tokens"] == 1500          # num_predict ignored, fixed budget
    assert p["reasoning_effort"] == "low"   # think=False mapping
    assert p["temperature"] == 0
    assert "response_format" not in p       # server ignores it; not sent
    assert "keep_alive" not in p            # Ollama-specific; not sent
    assert "num_predict" not in p
    assert captured["timeout_s"] == 180.0


def test_generate_ignores_reasoning_content(monkeypatch):
    monkeypatch.setattr(openai_client, "_post_json", lambda *a, **k: _ok_response("answer"))
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert resp.text == "answer"
    assert "secret" not in resp.text


def test_generate_rejects_images_without_http_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("HTTP must not be called for images")

    monkeypatch.setattr(openai_client, "_post_json", boom)
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m", images_b64=["Zm9v"])
    assert not resp.success
    assert "vision" in (resp.error or "")


def test_generate_maps_http_error_to_llmresponse(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(openai_client, "_post_json", boom)
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert not resp.success
    assert "TimeoutError" in (resp.error or "")


def test_generate_empty_content_is_error(monkeypatch):
    monkeypatch.setattr(openai_client, "_post_json", lambda *a, **k: _ok_response("   "))
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert not resp.success
    assert "empty" in (resp.error or "")


def test_generate_error_body_is_error(monkeypatch):
    monkeypatch.setattr(
        openai_client, "_post_json",
        lambda *a, **k: {"error": {"message": "model not found", "type": "invalid_request_error"}},
    )
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert not resp.success
    assert "model not found" in (resp.error or "")


def test_generate_malformed_response_is_error(monkeypatch):
    monkeypatch.setattr(openai_client, "_post_json", lambda *a, **k: {"choices": []})
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert not resp.success


def test_generate_non_dict_json_body_is_error(monkeypatch):
    monkeypatch.setattr(openai_client, "_post_json", lambda *a, **k: [])
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert not resp.success
    assert "malformed" in (resp.error or "")
