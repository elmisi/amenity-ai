"""Ollama HTTP API client.

This module provides the OllamaBackend class for interacting with Ollama,
implementing the LLMBackend protocol from llm_backend.py.

Backward-compatible module-level functions are also provided.
"""
from __future__ import annotations

import base64
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


# Backward-compatible module-level functions
_default_backend: Optional[OllamaBackend] = None


def _get_backend(base_url: str) -> OllamaBackend:
    """Get or create a backend instance."""
    global _default_backend
    if _default_backend is None or _default_backend.base_url != base_url.rstrip("/"):
        _default_backend = OllamaBackend(base_url)
    return _default_backend


def generate(
    *,
    model: str,
    prompt: str,
    base_url: str = "http://localhost:11434",
    timeout_s: float = 120.0,
    images_b64: Optional[list[str]] = None,
    response_format: str | dict[str, Any] | None = None,
    think: bool | str | None = None,
    keep_alive: str | int | None = None,
    options: Optional[dict[str, Any]] = None,
) -> OllamaGenerateResult:
    """Generate a response from Ollama (backward-compatible function)."""
    backend = _get_backend(base_url)
    response = backend.generate(
        prompt=prompt,
        model=model,
        timeout_s=timeout_s,
        images_b64=images_b64,
        response_format=response_format,
        think=think,
        keep_alive=keep_alive,
        options=options,
    )
    return OllamaGenerateResult(
        response=response.text,
        model=response.model,
        done=response.done,
        error=response.error,
    )


def generate_with_image_file(
    *,
    model: str,
    prompt: str,
    image_path: str,
    base_url: str = "http://localhost:11434",
    timeout_s: float = 180.0,
) -> OllamaGenerateResult:
    """Generate a response using an image file (backward-compatible function)."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return generate(
        model=model,
        prompt=prompt,
        base_url=base_url,
        timeout_s=timeout_s,
        images_b64=[b64],
        keep_alive="5m",
    )
