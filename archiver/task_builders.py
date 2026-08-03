from __future__ import annotations

from typing import TYPE_CHECKING

from .analyzer import AnalysisConfig
from .model_selection import ROLE_FACTS, ROLE_VISION, rank_models

if TYPE_CHECKING:  # pragma: no cover
    from .concurrency import ConcurrencyLimiter
    from .discovery import DiscoveryResult
    from .settings import Settings
    from .taxonomy import Taxonomy


def _with_pin(candidates: tuple[str, ...], pinned: str) -> tuple[str, ...]:
    if not pinned or pinned == "auto":
        return candidates
    return (pinned, *tuple(c for c in candidates if c != pinned))


def build_analysis_config(
    *,
    settings: "Settings",
    discovery: "DiscoveryResult | None",
    taxonomy: "Taxonomy",
    limiter: "ConcurrencyLimiter | None" = None,
) -> AnalysisConfig:
    models = discovery.models if discovery else ()
    text_models = _with_pin(rank_models(models, ROLE_FACTS), settings.facts_model)
    vision_models = _with_pin(rank_models(models, ROLE_VISION), settings.vision_model)
    return AnalysisConfig(
        output_language=settings.output_language,
        taxonomy=taxonomy,
        text_models=text_models,
        vision_models=vision_models,
        filename_separator=settings.filename_separator,
        ocr_mode=settings.ocr_mode,
        provider_urls=dict(settings.providers),
        limiter=limiter,
    )
