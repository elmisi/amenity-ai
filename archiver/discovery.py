from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import json
from urllib.request import urlopen


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    available: bool
    details: str
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryResult:
    providers: tuple[ProviderInfo, ...]
    chosen_text: Optional[str] = None
    chosen_vision: Optional[str] = None
    notes: tuple[str, ...] = ()


def _get_json(url: str, *, timeout_s: float) -> dict:
    with urlopen(url, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _discover_ds4(base_url: str) -> ProviderInfo:
    from .llm_router import DS4_PREFIX

    url = base_url.rstrip("/") + "/v1/models"
    try:
        data = _get_json(url, timeout_s=2.5)
    except Exception as exc:
        return ProviderInfo(name="ds4", available=False, details=f"Not reachable ({type(exc).__name__})")

    models: list[str] = []
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        entries = []
    for entry in entries:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and model_id.strip():
            models.append(DS4_PREFIX + model_id.strip())

    details = "OK" if models else "OK (no models listed)"
    return ProviderInfo(name="ds4", available=True, details=details, models=tuple(models))


def _discover_ollama(base_url: str) -> ProviderInfo:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        data = _get_json(url, timeout_s=2.5)
    except Exception as exc:
        return ProviderInfo(name="ollama", available=False, details=f"Not reachable ({type(exc).__name__})")

    models: list[str] = []
    entries = data.get("models") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        entries = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, str) and name.strip():
            models.append(name.strip())

    details = "OK" if models else "OK (no models listed)"
    return ProviderInfo(name="ollama", available=True, details=details, models=tuple(models))


def discover_providers(
    *,
    ollama_base_url: str = "http://localhost:11434",
    ds4_base_url: str = "",
) -> DiscoveryResult:
    providers: list[ProviderInfo] = []
    notes: list[str] = []

    ollama = _discover_ollama(ollama_base_url.strip() or "http://localhost:11434")
    providers.append(ollama)

    ds4 = None
    if ds4_base_url.strip():
        ds4 = _discover_ds4(ds4_base_url.strip())
        providers.append(ds4)

    chosen_text = None
    chosen_vision = None
    if ds4 is not None and ds4.available and ds4.models:
        chosen_text = "ds4"
    elif ollama.available and ollama.models:
        chosen_text = "ollama"
    elif ollama.available and not ollama.models:
        notes.append("Ollama has no models: run 'ollama pull <model>'.")

    return DiscoveryResult(
        providers=tuple(providers),
        chosen_text=chosen_text,
        chosen_vision=chosen_vision,
        notes=tuple(notes),
    )
