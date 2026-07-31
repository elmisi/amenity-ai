from __future__ import annotations

from archiver.discovery import DiscoveryResult, ProviderInfo
from archiver.model_selection import pick_model_candidates


def _discovery(ollama_models=(), ds4_models=()):
    providers = []
    if ollama_models:
        providers.append(ProviderInfo(name="ollama", available=True, details="OK", models=tuple(ollama_models)))
    if ds4_models:
        providers.append(ProviderInfo(name="ds4", available=True, details="OK", models=tuple(ds4_models)))
    return DiscoveryResult(providers=tuple(providers))


def test_ds4_models_lead_text_candidates():
    text, vision = pick_model_candidates(
        _discovery(
            ollama_models=("gemma3:1b", "moondream:latest", "qwen2.5:3b-instruct"),
            ds4_models=("ds4:deepseek-v4-flash", "ds4:deepseek-v4-pro"),
        )
    )
    assert text[0] == "ds4:deepseek-v4-flash"
    assert text[1] == "ds4:deepseek-v4-pro"
    assert "gemma3:1b" in text


def test_ds4_models_never_in_vision():
    text, vision = pick_model_candidates(
        _discovery(
            ollama_models=("moondream:latest", "llava:7b"),
            ds4_models=("ds4:deepseek-v4-flash",),
        )
    )
    assert all(not m.startswith("ds4:") for m in vision)
    assert "moondream:latest" in vision


def test_ollama_only_unchanged():
    text, vision = pick_model_candidates(_discovery(ollama_models=("gemma3:1b", "qwen2.5:3b-instruct")))
    assert text[0] == "gemma3:1b"
