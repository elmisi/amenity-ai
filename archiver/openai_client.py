"""Backend OpenAI-compatible, condiviso da ds4 e vLLM.

Implementa il protocollo LLMBackend su POST /v1/chat/completions. Legge
SOLO message.content, mai i campi di ragionamento: i modelli reasoning
riempiono un campo separato lasciando content a null finché non hanno
finito, e prenderlo per risposta significherebbe archiviare il monologo
del modello invece del suo output.

Le differenze fra provider dello stesso tipo (oggi: solo ds4 accetta
reasoning_effort) vivono nel registry, non qui.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Optional
from urllib.request import Request, urlopen

from .llm_backend import BaseLLMBackend, LLMResponse
from .providers import ProviderSpec

MAX_TOKENS = 8000


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


def mime_from_b64(b64: str) -> str:
    """Deduce il MIME dai magic bytes; default png se illeggibile."""
    try:
        head = base64.b64decode(b64[:32] + "==", validate=False)
    except Exception:
        return "image/png"
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"GIF8"):
        return "image/gif"
    return "image/png"


class OpenAICompatBackend(BaseLLMBackend):
    """Backend per qualunque server che parli l'API chat-completions di OpenAI.

    Usage:
        backend = OpenAICompatBackend(url, provider_by_name("vllm"))
        response = backend.generate(prompt="Hello", model="qwen3.6-27b")
    """

    def __init__(self, base_url: str, spec: Optional[ProviderSpec] = None):
        super().__init__(base_url)
        self.spec = spec

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
        max_model_len: Optional[int] = None,
    ) -> LLMResponse:
        if images_b64:
            content: Any = [{"type": "text", "text": prompt}]
            for b64 in images_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_from_b64(b64)};base64,{b64}"},
                    }
                )
        else:
            content = prompt

        max_tokens = MAX_TOKENS
        if isinstance(max_model_len, int) and 0 < max_model_len < MAX_TOKENS:
            max_tokens = max_model_len

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_tokens": max_tokens,
        }
        if think is False and self.spec is not None and self.spec.thinking_off:
            payload.update(self.spec.thinking_off)
        temperature = (options or {}).get("temperature")
        if isinstance(temperature, (int, float)):
            payload["temperature"] = temperature
        # response_format non è imposto dal server; keep_alive e num_predict
        # sono specifici di Ollama. La forma JSON è garantita dai prompt più
        # il normalizer e la riparazione JSON già esistenti.

        try:
            data = _post_json(f"{self.base_url}/v1/chat/completions", payload, timeout_s=timeout_s)
        except Exception as exc:
            return LLMResponse(text="", error=f"{type(exc).__name__}: {exc}", done=False)

        if not isinstance(data, dict):
            return LLMResponse(text="", error="openai-compat: malformed response", done=False)

        err = data.get("error")
        if err:
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return LLMResponse(text="", error=str(msg), done=False)

        try:
            choice = data["choices"][0]
            content_out = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return LLMResponse(text="", error="openai-compat: malformed response", done=False)

        if choice.get("finish_reason") == "length":
            return LLMResponse(text="", error="openai-compat: output truncated by max_tokens", done=False)
        if not content_out.strip():
            return LLMResponse(text="", error="openai-compat: empty content", done=False)
        return LLMResponse(text=content_out, model=data.get("model") or model, done=True)
