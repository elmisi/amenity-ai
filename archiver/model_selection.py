"""Ordering of the candidate models, by role.

Replaces the three hardcoded preference lists that used to live in
model_selection, task_builders and app. The order comes from real metadata;
the curated list only breaks ties inside a size band.

Criteria, in order:
  1. provider priority       (vllm > ollama > ds4)
  2. size band               (per role)
  3. position in CURATED_BIAS
  4. full id, alphabetical
"""
from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING

from .capabilities import CAP_COMPLETION, CAP_VISION
from .providers import provider_priority, split_model_id

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import ModelInfo

ROLE_FACTS = "facts"
ROLE_CLASSIFY = "classify"
ROLE_VISION = "vision"

ROLE_CAPABILITY = {
    ROLE_FACTS: CAP_COMPLETION,
    ROLE_CLASSIFY: CAP_COMPLETION,
    ROLE_VISION: CAP_VISION,
}

# Band edges, in billions of parameters: indices 0..4.
_BUCKET_EDGES = (2.0, 5.0, 9.0, 20.0)
_CLASSIFY_TARGET_BUCKET = 2  # the 5-9B band
_UNKNOWN_BUCKET_KEY = 99

# Models already tried in the past. It orders ONLY within a band, so being
# tuned on more limited hardware does not make it harmful.
# Maintained by hand, in dedicated passes: no auto-benchmarking.
CURATED_BIAS: tuple[str, ...] = (
    "ds4:deepseek-v4-flash",
    "ds4:deepseek-v4-pro",
    "qwen3.6-27b",
    "qwen3:8b",
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
    "moondream:latest",
    "llava:7b",
    "minicpm-v:latest",
    "bakllava:latest",
)


def size_bucket(size_b: Optional[float]) -> Optional[int]:
    if size_b is None:
        return None
    for index, edge in enumerate(_BUCKET_EDGES):
        if size_b < edge:
            return index
    return len(_BUCKET_EDGES)


def _bucket_key(model: "ModelInfo", role: str) -> int:
    bucket = size_bucket(model.parameter_size_b)
    if bucket is None:
        return _UNKNOWN_BUCKET_KEY
    if role == ROLE_CLASSIFY:
        return abs(bucket - _CLASSIFY_TARGET_BUCKET)
    return bucket


def _curated_key(model_id: str) -> int:
    if model_id in CURATED_BIAS:
        return CURATED_BIAS.index(model_id)
    _, bare = split_model_id(model_id)
    if bare in CURATED_BIAS:
        return CURATED_BIAS.index(bare)
    return len(CURATED_BIAS)


def rank_models(models: Sequence["ModelInfo"], role: str) -> tuple[str, ...]:
    required = ROLE_CAPABILITY.get(role, CAP_COMPLETION)
    eligible = [model for model in models if required in model.capabilities]
    eligible.sort(
        key=lambda model: (
            provider_priority(model.provider),
            _bucket_key(model, role),
            _curated_key(model.id),
            model.id,
        )
    )
    return tuple(model.id for model in eligible)
