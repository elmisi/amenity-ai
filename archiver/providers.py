"""Registry of the LLM providers.

The single source of truth for names, prefixes, priority and the ways
providers differ. The declaration order of PROVIDERS IS the priority the
ranking uses: vllm holds up under contention and will scale once scanning
runs in parallel; ollama is always there; ds4 serves one request at a
time, so a long scan would monopolise it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

KIND_OLLAMA = "ollama"
KIND_OPENAI_COMPAT = "openai_compat"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: str
    prefix: str
    default_url: str = ""
    # Fields to add to the payload to switch reasoning off. Every server has
    # its own lever and silently ignores the others', so a single boolean is
    # not enough: what matters is WHICH key to send.
    thinking_off: Mapping[str, Any] = field(default_factory=dict)
    supports_install: bool = False


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        "vllm",
        KIND_OPENAI_COMPAT,
        "vllm:",
        thinking_off={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    ProviderSpec(
        "ollama",
        KIND_OLLAMA,
        "ollama:",
        default_url="http://localhost:11434",
        supports_install=True,
    ),
    ProviderSpec(
        "ds4",
        KIND_OPENAI_COMPAT,
        "ds4:",
        thinking_off={"reasoning_effort": "low"},
    ),
)

PROVIDER_NAMES: tuple[str, ...] = tuple(p.name for p in PROVIDERS)

# Ids with no known prefix come from configs or caches written before
# 0.12.0, when a bare id meant Ollama.
_LEGACY_SPEC = next(p for p in PROVIDERS if p.name == "ollama")


def provider_by_name(name: str) -> Optional[ProviderSpec]:
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None


def provider_priority(name: str) -> int:
    for index, spec in enumerate(PROVIDERS):
        if spec.name == name:
            return index
    return len(PROVIDERS)


def split_model_id(model_id: str) -> tuple[ProviderSpec, str]:
    """Split prefix from bare id by matching against the KNOWN prefixes.

    It does not split on the first ':' it meets: "ollama:qwen3:8b" must give
    ("ollama", "qwen3:8b"), and "qwen3:8b" must not invent a "qwen3" provider.
    """
    for spec in PROVIDERS:
        if model_id.startswith(spec.prefix):
            return spec, model_id[len(spec.prefix):]
    return _LEGACY_SPEC, model_id


def join_model_id(provider_name: str, bare_id: str) -> str:
    spec = provider_by_name(provider_name)
    if spec is None:
        raise KeyError(f"unknown provider: {provider_name}")
    return spec.prefix + bare_id


def default_provider_urls() -> dict[str, str]:
    return {spec.name: spec.default_url for spec in PROVIDERS}
