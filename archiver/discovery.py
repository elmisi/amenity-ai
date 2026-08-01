from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    available: bool
    details: str
    models: tuple[str, ...] = ()
    command: Optional[str] = None


@dataclass(frozen=True)
class DiscoveryResult:
    providers: tuple[ProviderInfo, ...]
    chosen_text: Optional[str] = None
    chosen_vision: Optional[str] = None
    notes: tuple[str, ...] = ()


def _run(cmd: list[str], timeout_s: float = 2.5) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )


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
    for entry in entries or []:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and model_id.strip():
            models.append(DS4_PREFIX + model_id.strip())

    details = "OK" if models else "OK (no models listed)"
    return ProviderInfo(name="ds4", available=True, details=details, models=tuple(models))


def _discover_ollama() -> ProviderInfo:
    path = shutil.which("ollama")
    if not path:
        return ProviderInfo(name="ollama", available=False, details="Not found in PATH")

    proc = _run(["ollama", "list"], timeout_s=3.5)
    if proc.returncode != 0:
        details = proc.stderr.strip() or "Comando presente ma non risponde"
        return ProviderInfo(name="ollama", available=False, details=details, command=path)

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    models: list[str] = []
    for ln in lines[1:]:
        model = ln.split()[0].strip()
        if model:
            models.append(model)

    details = "OK"
    if not models:
        details = "OK (no models listed)"

    return ProviderInfo(
        name="ollama",
        available=True,
        details=details,
        models=tuple(models),
        command=path,
    )

def discover_providers(*, ds4_base_url: str = "") -> DiscoveryResult:
    providers: list[ProviderInfo] = []
    notes: list[str] = []

    ollama = _discover_ollama()
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
        notes.append("Ollama is installed but has no models: run 'ollama pull <model>'.")

    return DiscoveryResult(
        providers=tuple(providers),
        chosen_text=chosen_text,
        chosen_vision=chosen_vision,
        notes=tuple(notes),
    )
