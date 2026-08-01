from __future__ import annotations

import json
from pathlib import Path

from archiver.config import AppConfig, load_config, save_config
from archiver.discovery import DiscoveryResult, ProviderInfo
from archiver.settings import Settings
from archiver.setup_logic import app_config_from_settings
from archiver.task_builders import build_analysis_config
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)


def test_config_roundtrip_ds4_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_config(AppConfig(ds4_base_url="http://localhost:8000"))
    written = json.loads((tmp_path / "amenity-stuff" / "config.json").read_text())
    assert written["ds4_base_url"] == "http://localhost:8000"
    assert load_config().ds4_base_url == "http://localhost:8000"


def test_config_default_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert load_config().ds4_base_url == ""


def test_app_config_from_settings_carries_ds4():
    settings = Settings(
        source_root=Path("."),
        archive_root=Path("./ARCHIVE"),
        ds4_base_url="http://localhost:8000",
    )
    assert app_config_from_settings(settings).ds4_base_url == "http://localhost:8000"


def test_build_analysis_config_threads_ds4_and_prefers_flash():
    settings = Settings(
        source_root=Path("."),
        archive_root=Path("./ARCHIVE"),
        ds4_base_url="http://localhost:8000",
    )
    discovery = DiscoveryResult(
        providers=(
            ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",)),
            ProviderInfo(name="ds4", available=True, details="OK",
                         models=("ds4:deepseek-v4-flash", "ds4:deepseek-v4-pro")),
        )
    )
    cfg = build_analysis_config(settings=settings, discovery=discovery, taxonomy=_TAXONOMY)
    assert cfg.ds4_base_url == "http://localhost:8000"
    assert cfg.text_models[0] == "ds4:deepseek-v4-flash"


def test_config_roundtrip_ollama_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_config(AppConfig(ollama_base_url="http://ollama-box:11434"))
    written = json.loads((tmp_path / "amenity-stuff" / "config.json").read_text())
    assert written["ollama_base_url"] == "http://ollama-box:11434"
    assert load_config().ollama_base_url == "http://ollama-box:11434"


def test_config_ollama_default_is_localhost(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert load_config().ollama_base_url == "http://localhost:11434"


def test_config_ollama_blank_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg_dir = tmp_path / "amenity-stuff"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(json.dumps({"ollama_base_url": "   "}))
    assert load_config().ollama_base_url == "http://localhost:11434"


def test_app_config_from_settings_carries_ollama_url():
    settings = Settings(
        source_root=Path("."),
        archive_root=Path("./ARCHIVE"),
        ollama_base_url="http://ollama-box:11434",
    )
    assert app_config_from_settings(settings).ollama_base_url == "http://ollama-box:11434"


def test_build_analysis_config_threads_ollama_url():
    settings = Settings(
        source_root=Path("."),
        archive_root=Path("./ARCHIVE"),
        ollama_base_url="http://ollama-box:11434",
    )
    discovery = DiscoveryResult(
        providers=(ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",)),)
    )
    cfg = build_analysis_config(settings=settings, discovery=discovery, taxonomy=_TAXONOMY)
    assert cfg.ollama_base_url == "http://ollama-box:11434"
