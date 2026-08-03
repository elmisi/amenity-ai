from pathlib import Path

from archiver.concurrency import clamp_limit
from archiver.providers import PROVIDERS
from archiver.settings_screen import SettingsResult


def test_result_carries_the_limits():
    result = SettingsResult(
        output_language="auto",
        taxonomies={},
        facts_model="auto",
        classify_model="auto",
        vision_model="auto",
        filename_separator="space",
        ocr_mode="balanced",
        undated_folder_name="undated",
        archive_root=Path("/tmp/arc"),
        providers={"vllm": "http://example.invalid"},
        provider_concurrency={"vllm": 4, "ollama": 1, "ds4": 1},
    )
    assert result.provider_concurrency["vllm"] == 4


def test_an_emptied_field_falls_back_to_the_registry_default():
    for spec in PROVIDERS:
        assert clamp_limit("", default=spec.max_concurrency) == spec.max_concurrency
