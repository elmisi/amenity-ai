"""OpenAI-compatible chat-completions backend ("ds4").

Implements the LLMBackend protocol over POST /v1/chat/completions.
Targets local reasoning models: reads ONLY message.content (never
reasoning_content), forces low reasoning effort and a fixed max_tokens
budget large enough that the reasoning phase cannot swallow the answer.
"""
from __future__ import annotations

import json
from typing import Any, Optional
from urllib.request import Request, urlopen

from .llm_backend import BaseLLMBackend, LLMResponse

_MAX_TOKENS = 8000


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


class Ds4Backend(BaseLLMBackend):
    """OpenAI-compatible LLM backend (text only).

    Usage:
        backend = Ds4Backend("http://localhost:8000")
        response = backend.generate(prompt="Hello", model="deepseek-v4-flash")
    """

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        timeout_s: float = 120.0,
        images_b64: Optional[list[str]] = None,
        response_format: str | dict[str, Any] | None = None,
        think: bool | str | None = None,
        keep_alive: str | int | None = None,
        options: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        if images_b64:
            return LLMResponse(text="", error="ds4: vision not supported", done=False)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": _MAX_TOKENS,
        }
        if think is False:
            payload["reasoning_effort"] = "low"
        temperature = (options or {}).get("temperature")
        if isinstance(temperature, (int, float)):
            payload["temperature"] = temperature
        # response_format is accepted but not enforced by the server; keep_alive
        # and num_predict are Ollama-specific. None of them are sent: JSON shape
        # is guaranteed by the prompts + the existing normalizer/JSON repair.

        try:
            data = _post_json(f"{self.base_url}/v1/chat/completions", payload, timeout_s=timeout_s)
        except Exception as exc:
            return LLMResponse(text="", error=f"{type(exc).__name__}: {exc}", done=False)

        if not isinstance(data, dict):
            return LLMResponse(text="", error="ds4: malformed response", done=False)

        err = data.get("error")
        if err:
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return LLMResponse(text="", error=str(msg), done=False)

        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return LLMResponse(text="", error="ds4: malformed response", done=False)
        finish_reason = None
        try:
            finish_reason = data["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            pass
        if finish_reason == "length":
            return LLMResponse(text="", error="ds4: output truncated by max_tokens", done=False)
        if not content.strip():
            return LLMResponse(text="", error="ds4: empty content", done=False)
        return LLMResponse(text=content, model=data.get("model") or model, done=True)
