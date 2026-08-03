from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from archiver.settings_screen import SettingsResult


def test_result_carries_a_providers_mapping():
    names = {f.name for f in fields(SettingsResult)}
    assert "providers" in names
    assert "ds4_base_url" not in names
    assert "ollama_base_url" not in names
    assert "vision_model_fallback" not in names


def test_result_is_constructible_with_three_providers():
    result = SettingsResult(
        output_language="it",
        taxonomies={},
        facts_model="auto",
        classify_model="auto",
        vision_model="auto",
        filename_separator="space",
        ocr_mode="balanced",
        undated_folder_name="undated",
        archive_root=Path("/tmp/archive"),
        providers={"ollama": "http://a.invalid", "vllm": "", "ds4": ""},
        provider_concurrency={"ollama": 1, "vllm": 4, "ds4": 1},
    )
    assert set(result.providers) == {"ollama", "vllm", "ds4"}
    assert set(result.provider_concurrency) == {"ollama", "vllm", "ds4"}
