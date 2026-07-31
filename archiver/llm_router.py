"""Route LLM calls to the right backend based on the model-id prefix.

Convention: model ids starting with "ds4:" go to the OpenAI-compatible
endpoint configured via Settings.ds4_base_url (prefix stripped before the
HTTP call); every other model id goes to Ollama. The prefix travels with
the model id everywhere (candidates, settings, cache model_used, UI), so
no extra state is needed to know a model's provider.
"""
from __future__ import annotations

from typing import Any, Optional

from .ollama_client import OllamaGenerateResult
from .ollama_client import generate as _ollama_generate
from .ollama_client import generate_with_image_file as _ollama_generate_with_image_file
from .openai_client import Ds4Backend

DS4_PREFIX = "ds4:"


def is_ds4_model(model: str) -> bool:
    return model.startswith(DS4_PREFIX)


def generate(
    *,
    model: str,
    prompt: str,
    base_url: str = "http://localhost:11434",
    ds4_base_url: str = "",
    timeout_s: float = 120.0,
    images_b64: Optional[list[str]] = None,
    response_format: str | dict[str, Any] | None = None,
    think: bool | str | None = None,
    keep_alive: str | int | None = None,
    options: Optional[dict[str, Any]] = None,
) -> OllamaGenerateResult:
    if is_ds4_model(model):
        if not ds4_base_url:
            return OllamaGenerateResult(response="", error="ds4 endpoint not configured", done=False)
        resp = Ds4Backend(ds4_base_url).generate(
            prompt=prompt,
            model=model[len(DS4_PREFIX):],
            timeout_s=timeout_s,
            images_b64=images_b64,
            response_format=response_format,
            think=think,
            keep_alive=keep_alive,
            options=options,
        )
        return OllamaGenerateResult(response=resp.text, model=model, done=resp.done, error=resp.error)
    return _ollama_generate(
        model=model,
        prompt=prompt,
        base_url=base_url,
        timeout_s=timeout_s,
        images_b64=images_b64,
        response_format=response_format,
        think=think,
        keep_alive=keep_alive,
        options=options,
    )


def generate_with_image_file(
    *,
    model: str,
    prompt: str,
    image_path: str,
    base_url: str = "http://localhost:11434",
    ds4_base_url: str = "",
    timeout_s: float = 180.0,
) -> OllamaGenerateResult:
    if is_ds4_model(model):
        return OllamaGenerateResult(response="", error="ds4: vision not supported", done=False)
    return _ollama_generate_with_image_file(
        model=model,
        prompt=prompt,
        image_path=image_path,
        base_url=base_url,
        timeout_s=timeout_s,
    )
