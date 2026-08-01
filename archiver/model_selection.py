from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .llm_router import DS4_PREFIX

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import DiscoveryResult


_DS4_TEXT_PREFER = (
    "ds4:deepseek-v4-flash",
    "ds4:deepseek-v4-pro",
)

_TEXT_PREFER = _DS4_TEXT_PREFER + (
    "gemma3:1b",
    "qwen2.5:3b-instruct",
    "phi4-mini:latest",
    "phi4-mini",
    "qwen3:4b",
    "qwen3.5:4b",
    "ministral-3:3b",
    "gemma2:2b",
    "qwen2.5:7b",
    "mistral:latest",
    "gemma3:latest",
)

_VISION_PREFER = (
    "moondream:latest",
    "gemma3:latest",
    "llava:latest",
    "llava:7b",
    "minicpm-v:latest",
    "bakllava:latest",
)


def _is_vision_model(model_name: str) -> bool:
    ml = model_name.lower()
    if any(token in ml for token in ("llava", "moondream", "minicpm", "bakllava")):
        return True
    if "vision" in ml:
        return True
    if ml.startswith("gemma3:"):
        return not any(token in ml for token in ("270m", "1b"))
    return False


def _is_text_candidate(model_name: str) -> bool:
    ml = model_name.lower()
    if any(token in ml for token in ("embed", "embedding", "whisper", "tts")):
        return False
    return True


def _order_candidates(models: list[str], preferred: tuple[str, ...]) -> list[str]:
    ordered = [model for model in preferred if model in models]
    ordered.extend(model for model in models if model not in ordered)
    return ordered


def pick_model_candidates(discovery: "DiscoveryResult | None") -> tuple[tuple[str, ...], tuple[str, ...]]:
    models: list[str] = []
    if discovery:
        for p in discovery.providers:
            if p.name in ("ollama", "ds4") and p.available and p.models:
                models.extend(p.models)

    if not models:
        return (), ()

    text_candidates = [
        model
        for model in models
        if _is_text_candidate(model) and not (_is_vision_model(model) and not model.lower().startswith(("gemma3:", "ministral-3:")))
    ]
    vision_candidates = [model for model in models if _is_vision_model(model) and not model.startswith(DS4_PREFIX)]

    # Keep newer generic names such as `qwen3:4b` and `phi4-mini:latest` eligible even
    # when they don't advertise themselves with `-instruct` or `-chat`.
    text_candidates = _order_candidates(text_candidates, _TEXT_PREFER)
    vision_candidates = _order_candidates(vision_candidates, _VISION_PREFER)

    # If discovery returns only generic model names, preserve a predictable ordering.
    text_candidates = sorted(
        text_candidates,
        key=lambda model: (
            model not in _TEXT_PREFER,
            not re.search(r":(?:270m|1b|2b|3b|4b)\b", model.lower()),
            model.lower(),
        ),
    )
    text_candidates = _order_candidates(text_candidates, _TEXT_PREFER)

    return tuple(text_candidates[:6]), tuple(vision_candidates[:4])
