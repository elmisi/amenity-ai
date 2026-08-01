from __future__ import annotations

import json

from pathlib import Path

from archiver.config import AppConfig, load_config, migrate_model_id, save_config
from archiver.settings import Settings
from archiver.setup_logic import app_config_from_settings


def _write(tmp_path, monkeypatch, data):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "amenity-stuff" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_defaults_have_three_providers_and_no_real_hostnames():
    cfg = AppConfig()
    assert set(cfg.providers) == {"ollama", "vllm", "ds4"}
    assert cfg.providers["ollama"] == "http://localhost:11434"
    assert cfg.providers["vllm"] == ""
    assert cfg.providers["ds4"] == ""


def test_flat_urls_are_migrated_into_the_providers_mapping(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {
        "ollama_base_url": "http://box.invalid:11434",
        "ds4_base_url": "http://box.invalid:9000",
    })
    cfg = load_config()
    assert cfg.providers["ollama"] == "http://box.invalid:11434"
    assert cfg.providers["ds4"] == "http://box.invalid:9000"
    assert cfg.providers["vllm"] == ""


def test_bare_model_ids_are_migrated_to_the_ollama_prefix():
    assert migrate_model_id("qwen3:8b") == "ollama:qwen3:8b"
    assert migrate_model_id("ds4:deepseek-v4-flash") == "ds4:deepseek-v4-flash"
    assert migrate_model_id("auto") == "auto"
    assert migrate_model_id("") == "auto"


def test_pinned_models_in_an_old_config_are_migrated(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {
        "facts_model": "gemma3:1b",
        "classify_model": "ds4:deepseek-v4-flash",
        "vision_model": "auto",
    })
    cfg = load_config()
    assert cfg.facts_model == "ollama:gemma3:1b"
    assert cfg.classify_model == "ds4:deepseek-v4-flash"
    assert cfg.vision_model == "auto"


def test_legacy_vision_model_fallback_is_dropped(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"vision_model_fallback": "llava:7b"})
    cfg = load_config()
    assert not hasattr(cfg, "vision_model_fallback")


def test_legacy_text_model_migration_still_works(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"text_model": "gemma3:1b"})
    cfg = load_config()
    assert cfg.facts_model == "ollama:gemma3:1b"
    assert cfg.classify_model == "ollama:gemma3:1b"


def test_new_format_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_config(AppConfig(providers={"ollama": "http://a.invalid", "vllm": "http://b.invalid", "ds4": ""}))
    cfg = load_config()
    assert cfg.providers["vllm"] == "http://b.invalid"


def test_unknown_provider_keys_are_ignored(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"providers": {"ollama": "http://a.invalid", "bogus": "x"}})
    cfg = load_config()
    assert set(cfg.providers) == {"ollama", "vllm", "ds4"}


def test_missing_file_yields_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert load_config().providers["ollama"] == "http://localhost:11434"


def test_a_blank_url_on_disk_falls_back_to_the_default(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"providers": {"ollama": "   "}})
    assert load_config().providers["ollama"] == "http://localhost:11434"


def test_settings_carry_providers_back_into_the_saved_config():
    settings = Settings(
        source_root=Path("."),
        archive_root=Path("./ARCHIVE"),
        providers={"ollama": "http://box.invalid:11434", "vllm": "http://box.invalid:8000", "ds4": ""},
    )
    saved = app_config_from_settings(settings)
    assert saved.providers["vllm"] == "http://box.invalid:8000"
    assert saved.providers["ollama"] == "http://box.invalid:11434"
