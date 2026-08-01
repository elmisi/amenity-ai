from __future__ import annotations

from pathlib import Path

from archiver.settings_screen import SettingsResult


def test_settings_result_carries_ds4_base_url():
    r = SettingsResult(
        output_language="auto",
        taxonomies={},
        facts_model="auto",
        classify_model="auto",
        vision_model="auto",
        vision_model_fallback="none",
        filename_separator="space",
        ocr_mode="balanced",
        undated_folder_name="undated",
        archive_root=Path("./ARCHIVE"),
        ds4_base_url="http://localhost:8000",
    )
    assert r.ds4_base_url == "http://localhost:8000"
