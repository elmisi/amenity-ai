from __future__ import annotations

from typing import TYPE_CHECKING

from .analyzer import AnalysisConfig
from .model_selection import pick_model_candidates

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import DiscoveryResult
    from .settings import Settings
    from .taxonomy import Taxonomy


def build_analysis_config(*, settings: "Settings", discovery: "DiscoveryResult | None", taxonomy: "Taxonomy") -> AnalysisConfig:
    text_models, vision_models = pick_model_candidates(discovery)
    # Facts extraction is latency-sensitive: prefer a smaller model when available.
    if (not settings.facts_model) or settings.facts_model == "auto":
        prefer_fast = (
            "ds4:deepseek-v4-flash",
            "ds4:deepseek-v4-pro",
            "gemma3:1b",
            "qwen2.5:3b-instruct",
            "phi4-mini:latest",
            "phi4-mini",
            "qwen3:4b",
            "qwen3.5:4b",
            "ministral-3:3b",
            "gemma2:2b",
            "qwen2.5:7b",
        )
        ordered = [m for m in prefer_fast if m in text_models] + [m for m in text_models if m not in prefer_fast]
        text_models = tuple(ordered)
    if settings.facts_model and settings.facts_model != "auto":
        text_models = (settings.facts_model, *tuple(m for m in text_models if m != settings.facts_model))
    if settings.vision_model and settings.vision_model != "auto":
        vision_models = (settings.vision_model, *tuple(m for m in vision_models if m != settings.vision_model))

    # Add fallback vision model if configured
    if settings.vision_model_fallback and settings.vision_model_fallback not in ("none", "auto"):
        # Add explicit fallback model after primary
        fallback = settings.vision_model_fallback
        if fallback not in vision_models:
            vision_models = (*vision_models, fallback)
    elif settings.vision_model_fallback == "auto" and len(vision_models) < 2:
        # Auto fallback: add llava:7b if not already in the list
        for candidate in ("llava:7b", "llava:13b", "minicpm-v"):
            if candidate not in vision_models:
                vision_models = (*vision_models, candidate)
                break

    return AnalysisConfig(
        output_language=settings.output_language,
        taxonomy=taxonomy,
        text_models=text_models,
        vision_models=vision_models,
        filename_separator=settings.filename_separator,
        ocr_mode=settings.ocr_mode,
        ds4_base_url=settings.ds4_base_url,
        ollama_base_url=settings.ollama_base_url,
    )
