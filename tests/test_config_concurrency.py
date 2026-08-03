import json
from pathlib import Path

from archiver.config import AppConfig, load_config, save_config
from archiver.settings import Settings


def test_a_config_written_before_0_13_gets_the_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "amenity-stuff" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"providers": {"vllm": "http://example.invalid"}}))
    cfg = load_config()
    assert cfg.provider_concurrency == {"vllm": 4, "ollama": 1, "ds4": 1}


def test_stored_values_win_and_are_clamped(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "amenity-stuff" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"provider_concurrency": {"vllm": 8, "ollama": 0, "ds4": 999}}))
    cfg = load_config()
    assert cfg.provider_concurrency == {"vllm": 8, "ollama": 1, "ds4": 16}


def test_unknown_provider_names_are_dropped():
    cfg = AppConfig(provider_concurrency={"vllm": 2, "ghost": 5})
    assert cfg.provider_concurrency == {"vllm": 2, "ollama": 1, "ds4": 1}


def test_settings_normalises_the_same_way():
    settings = Settings(
        source_root=Path("/tmp/src"),
        archive_root=Path("/tmp/arc"),
        provider_concurrency={"vllm": 6},
    )
    assert settings.provider_concurrency == {"vllm": 6, "ollama": 1, "ds4": 1}


def test_the_limits_survive_a_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_config(AppConfig(provider_concurrency={"vllm": 3}))
    assert load_config().provider_concurrency["vllm"] == 3
