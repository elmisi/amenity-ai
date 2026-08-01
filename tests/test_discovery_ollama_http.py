from __future__ import annotations

from archiver import discovery
from archiver.discovery import ProviderInfo, _discover_ollama, discover_providers


def test_discover_ollama_parses_api_tags(monkeypatch):
    captured = {}

    def fake_get(url, *, timeout_s):
        captured["url"] = url
        captured["timeout_s"] = timeout_s
        return {"models": [{"name": "gemma3:1b"}, {"name": "moondream:latest"}]}

    monkeypatch.setattr(discovery, "_get_json", fake_get)
    info = _discover_ollama("http://localhost:11434/")

    assert info.name == "ollama"
    assert info.available
    assert info.models == ("gemma3:1b", "moondream:latest")
    assert info.details == "OK"
    assert captured["url"] == "http://localhost:11434/api/tags"  # trailing slash normalized
    assert captured["timeout_s"] == 2.5


def test_discover_ollama_uses_configured_remote_url(monkeypatch):
    captured = {}

    def fake_get(url, *, timeout_s):
        captured["url"] = url
        return {"models": []}

    monkeypatch.setattr(discovery, "_get_json", fake_get)
    info = _discover_ollama("http://ollama-box:11434")

    assert captured["url"] == "http://ollama-box:11434/api/tags"
    assert info.available
    assert info.details == "OK (no models listed)"


def test_discover_ollama_server_down(monkeypatch):
    def boom(url, *, timeout_s):
        raise OSError("connection refused")

    monkeypatch.setattr(discovery, "_get_json", boom)
    info = _discover_ollama("http://localhost:11434")

    assert info.name == "ollama"
    assert not info.available
    assert info.models == ()
    assert "Not reachable" in info.details


def test_discover_ollama_malformed_payload(monkeypatch):
    monkeypatch.setattr(discovery, "_get_json", lambda url, *, timeout_s: ["not", "a", "dict"])
    info = _discover_ollama("http://localhost:11434")
    assert info.available
    assert info.models == ()


def test_discover_ollama_skips_unnamed_entries(monkeypatch):
    monkeypatch.setattr(
        discovery, "_get_json",
        lambda url, *, timeout_s: {"models": [{"name": "gemma3:1b"}, {"size": 12}, "junk", {"name": "  "}]},
    )
    info = _discover_ollama("http://localhost:11434")
    assert info.models == ("gemma3:1b",)


def test_discover_providers_passes_ollama_url(monkeypatch):
    captured = {}

    def fake_ollama(base_url):
        captured["base_url"] = base_url
        return ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",))

    monkeypatch.setattr(discovery, "_discover_ollama", fake_ollama)
    result = discover_providers(ollama_base_url="http://ollama-box:11434")

    assert captured["base_url"] == "http://ollama-box:11434"
    assert [p.name for p in result.providers] == ["ollama"]
    assert result.chosen_text == "ollama"


def test_discover_providers_defaults_to_localhost(monkeypatch):
    captured = {}

    def fake_ollama(base_url):
        captured["base_url"] = base_url
        return ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",))

    monkeypatch.setattr(discovery, "_discover_ollama", fake_ollama)
    discover_providers()
    assert captured["base_url"] == "http://localhost:11434"


def test_discover_ollama_non_list_models_does_not_raise(monkeypatch):
    monkeypatch.setattr(discovery, "_get_json", lambda url, *, timeout_s: {"models": 5})
    info = _discover_ollama("http://localhost:11434")
    assert info.available
    assert info.models == ()


def test_discover_ds4_non_list_data_does_not_raise(monkeypatch):
    from archiver.discovery import _discover_ds4

    monkeypatch.setattr(discovery, "_get_json", lambda url, *, timeout_s: {"data": 5})
    info = _discover_ds4("http://localhost:8000")
    assert info.available
    assert info.models == ()
