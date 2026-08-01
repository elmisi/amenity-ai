"""Instrada le chiamate LLM al backend giusto, in base al prefisso del model-id.

Convenzione: ogni model-id porta con sé il provider ("ollama:", "vllm:",
"ds4:"). Il prefisso viaggia ovunque — candidati, config, model_used in
cache, UI — quindi non serve altro stato per sapere da dove viene un
modello. Gli id senza prefisso noto vengono da config scritte prima della
0.12.0 e valgono come Ollama.

Il layer è stateless: un backend per chiamata, nessuno stato mutabile
condiviso, così una futura scansione parallela non dovrà toccarlo.
"""
from __future__ import annotations

import base64
from typing import Any, Mapping, Optional

from .llm_backend import LLMResponse
from .ollama_client import OllamaBackend, OllamaGenerateResult
from .openai_client import OpenAICompatBackend
from .providers import KIND_OLLAMA, split_model_id


def _to_legacy(response: LLMResponse, *, model: str) -> OllamaGenerateResult:
    return OllamaGenerateResult(
        response=response.text,
        model=model,
        done=response.done,
        error=response.error,
    )


def _resolve(model: str, provider_urls: Mapping[str, str]):
    spec, bare_id = split_model_id(model)
    url = (provider_urls.get(spec.name) or "").strip()
    if not url:
        return None, spec, bare_id
    if spec.kind == KIND_OLLAMA:
        return OllamaBackend(url), spec, bare_id
    return OpenAICompatBackend(url, spec), spec, bare_id


def generate(
    *,
    model: str,
    prompt: str,
    provider_urls: Mapping[str, str],
    timeout_s: float = 120.0,
    images_b64: Optional[list[str]] = None,
    response_format: str | dict[str, Any] | None = None,
    think: bool | str | None = None,
    keep_alive: str | int | None = None,
    options: Optional[dict[str, Any]] = None,
    max_model_len: Optional[int] = None,
) -> OllamaGenerateResult:
    backend, spec, bare_id = _resolve(model, provider_urls)
    if backend is None:
        return OllamaGenerateResult(
            response="", error=f"{spec.name}: endpoint non configurato", done=False
        )
    kwargs: dict[str, Any] = dict(
        prompt=prompt,
        model=bare_id,
        timeout_s=timeout_s,
        images_b64=images_b64,
        response_format=response_format,
        think=think,
        keep_alive=keep_alive,
        options=options,
    )
    if spec.kind != KIND_OLLAMA:
        kwargs["max_model_len"] = max_model_len
    return _to_legacy(backend.generate(**kwargs), model=model)


def generate_with_image_file(
    *,
    model: str,
    prompt: str,
    image_path: str,
    provider_urls: Mapping[str, str],
    timeout_s: float = 180.0,
) -> OllamaGenerateResult:
    with open(image_path, "rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("ascii")
    return generate(
        model=model,
        prompt=prompt,
        provider_urls=provider_urls,
        timeout_s=timeout_s,
        images_b64=[b64],
        keep_alive="5m",
    )
