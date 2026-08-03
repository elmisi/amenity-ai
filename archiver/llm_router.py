"""Route LLM calls to the right backend, based on the model-id prefix.

Convention: every model id carries its provider ("ollama:", "vllm:",
"ds4:"). The prefix travels everywhere — candidates, config, model_used in
the cache, the UI — so no extra state is needed to know where a model comes
from. Ids with no known prefix come from configs written before 0.12.0 and
count as Ollama.

The layer holds no state of its own: one backend per call, and the concurrency
limit arrives as an argument rather than living here, so a parallel scan needs
no coordination inside this module.
"""
from __future__ import annotations

import base64
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Mapping, Optional

from .llm_backend import LLMResponse
from .ollama_client import OllamaBackend, OllamaGenerateResult
from .openai_client import OpenAICompatBackend
from .providers import KIND_OLLAMA, split_model_id

if TYPE_CHECKING:  # pragma: no cover
    from .concurrency import ConcurrencyLimiter


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
    limiter: Optional["ConcurrencyLimiter"] = None,
) -> OllamaGenerateResult:
    backend, spec, bare_id = _resolve(model, provider_urls)
    if backend is None:
        return OllamaGenerateResult(
            response="", error=f"{spec.name}: endpoint not configured", done=False
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
    # The slot wraps the request and nothing else: building a prompt does not
    # occupy the server, and holding a slot while doing it would waste it.
    ctx = limiter.slot(spec.name) if limiter is not None else nullcontext()
    with ctx:
        response = backend.generate(**kwargs)
    return _to_legacy(response, model=model)


def generate_with_image_file(
    *,
    model: str,
    prompt: str,
    image_path: str,
    provider_urls: Mapping[str, str],
    timeout_s: float = 180.0,
    limiter: Optional["ConcurrencyLimiter"] = None,
) -> OllamaGenerateResult:
    with open(image_path, "rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("ascii")
    spec, _ = split_model_id(model)
    return generate(
        model=model,
        prompt=prompt,
        provider_urls=provider_urls,
        timeout_s=timeout_s,
        images_b64=[b64],
        keep_alive="5m",
        limiter=limiter,
        # A one-line caption has nothing to reason about, and a reasoning
        # model spends an order of magnitude more on it. On Ollama the flag
        # stays unset: not all of its vision models accept it, and the cost
        # of reasoning does not show up there.
        think=False if spec.kind != KIND_OLLAMA else None,
    )
