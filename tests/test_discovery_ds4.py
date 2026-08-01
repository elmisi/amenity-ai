from __future__ import annotations

from archiver import discovery
from archiver.discovery import DiscoveryResult, ProviderInfo, _discover_ds4, discover_providers


def test_discover_ds4_prefixes_model_ids(monkeypatch):
    monkeypatch.setattr(
        discovery, "_get_json",
        lambda url, *, timeout_s: {"object": "list", "data": [
            {"id": "deepseek-v4-flash", "object": "model"},
            {"id": "deepseek-v4-pro", "object": "model"},
        ]},
    )
    info = _discover_ds4("http://localhost:8000")
    assert info.name == "ds4"
    assert info.available
    assert info.models == ("ds4:deepseek-v4-flash", "ds4:deepseek-v4-pro")


def test_discover_ds4_server_down(monkeypatch):
    def boom(url, *, timeout_s):
        raise OSError("connection refused")

    monkeypatch.setattr(discovery, "_get_json", boom)
    info = _discover_ds4("http://localhost:8000")
    assert info.name == "ds4"
    assert not info.available
    assert info.models == ()


def test_discover_providers_skips_ds4_when_url_empty(monkeypatch):
    monkeypatch.setattr(
        discovery, "_discover_ollama",
        lambda base_url: ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",)),
    )

    def boom(base_url):
        raise AssertionError("ds4 must not be probed when url is empty")

    monkeypatch.setattr(discovery, "_discover_ds4", boom)
    result = discover_providers(ds4_base_url="")
    assert [p.name for p in result.providers] == ["ollama"]


def test_discover_providers_includes_ds4(monkeypatch):
    monkeypatch.setattr(
        discovery, "_discover_ollama",
        lambda base_url: ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",)),
    )
    monkeypatch.setattr(
        discovery, "_discover_ds4",
        lambda base_url: ProviderInfo(name="ds4", available=True, details="OK", models=("ds4:deepseek-v4-flash",)),
    )
    result = discover_providers(ds4_base_url="http://localhost:8000")
    assert [p.name for p in result.providers] == ["ollama", "ds4"]
