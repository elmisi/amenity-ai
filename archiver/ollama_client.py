"""Ollama HTTP API client.

This module provides the OllamaBackend class for interacting with Ollama,
implementing the LLMBackend protocol from llm_backend.py.

Everything goes through llm_router, which builds a backend per call: there is
no module-level state here for a parallel scan to contend over.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional
from urllib.request import Request, urlopen

from .llm_backend import BaseLLMBackend, LLMResponse


# Keep the old result class for backward compatibility
@dataclass(frozen=True)
class OllamaGenerateResult:
    """Legacy result class for backward compatibility."""

    response: str
    model: Optional[str] = None
    done: Optional[bool] = None
    error: Optional[str] = None


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    """Make a POST request with JSON payload."""
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


class OllamaBackend(BaseLLMBackend):
    """Ollama LLM backend implementation.

    Usage:
        backend = OllamaBackend("http://localhost:11434")
        response = backend.generate(prompt="Hello", model="qwen2.5:3b-instruct")
        if response.success:
            print(response.text)
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        super().__init__(base_url)

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
        """Generate a response from Ollama.

        Args:
            prompt: The prompt text.
            model: The model identifier (e.g., "qwen2.5:3b-instruct").
            timeout_s: Timeout in seconds.
            images_b64: Optional list of base64-encoded images for vision models.

        Returns:
            LLMResponse with the generated text or an error.
        """
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if images_b64:
            payload["images"] = images_b64
        if response_format is not None:
            payload["format"] = response_format
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if options:
            payload["options"] = options

        try:
            data = _post_json(url, payload, timeout_s=timeout_s)
            error = data.get("error") if isinstance(data.get("error"), str) else None
            if error is None and data.get("done_reason") == "length":
                # Hit the num_predict ceiling: the payload is cut mid-token and any
                # JSON in it is unparseable. Report it so the caller falls through to
                # the next candidate instead of "repairing" truncated garbage.
                error = "ollama: output truncated by num_predict"
            return LLMResponse(
                text="" if error else str(data.get("response", "")),
                model=data.get("model"),
                done=data.get("done", True) and error is None,
                error=error,
            )
        except Exception as exc:
            return LLMResponse(
                text="",
                error=f"{type(exc).__name__}: {exc}",
                done=False,
            )
