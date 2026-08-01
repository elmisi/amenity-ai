"""Registry dei provider LLM.

Unica fonte di verità su nomi, prefissi, priorità e differenze di
comportamento fra provider. L'ordine di dichiarazione di PROVIDERS È la
priorità usata dal ranking: vllm regge la contesa e scalerà quando la
scansione verrà parallelizzata; ollama è sempre disponibile; ds4 è
mutuamente esclusivo, quindi una scansione lunga lo monopolizzerebbe.
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
    # Campi da aggiungere al payload per spegnere il ragionamento. Ogni server
    # ha la sua leva e ignora silenziosamente quella degli altri, quindi un
    # flag booleano unico non basta: serve sapere QUALE chiave usare.
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

# Gli id senza prefisso noto vengono da config o cache scritte prima della
# 0.12.0, quando "nudo" significava Ollama.
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
    """Separa prefisso e id nudo confrontando con i prefissi NOTI.

    Non spezza sul primo ':' incontrato: "ollama:qwen3:8b" deve dare
    ("ollama", "qwen3:8b") e "qwen3:8b" non deve dare un provider "qwen3".
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
