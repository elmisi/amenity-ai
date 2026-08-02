"""Persistence of conclusive capability-probe results.

Without it, the knowledge that an OpenAI-compatible model is multimodal
would be lost on every restart and the fast path would keep trusting the
name heuristic, which is wrong in exactly that case.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_SEP = "|"


def probe_cache_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "amenity-stuff" / "probe_cache.json"


def _resolve(path: Optional[Path]) -> Path:
    return path if path is not None else probe_cache_path()


def _normalise_url(url: str) -> str:
    return url.strip().rstrip("/")


def load_probe_cache(path: Optional[Path] = None) -> dict[tuple[str, str], frozenset[str]]:
    target = _resolve(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[tuple[str, str], frozenset[str]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or _SEP not in key:
            continue
        url, _, bare_id = key.partition(_SEP)
        if not isinstance(value, list):
            continue
        caps = frozenset(v for v in value if isinstance(v, str) and v.strip())
        if url and bare_id and caps:
            out[(url, bare_id)] = caps
    return out


def _write(cache: dict[tuple[str, str], frozenset[str]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {f"{url}{_SEP}{bare}": sorted(caps) for (url, bare), caps in cache.items()}
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def save_probe_result(
    *,
    url: str,
    bare_id: str,
    capabilities: frozenset[str],
    path: Optional[Path] = None,
) -> None:
    target = _resolve(path)
    cache = load_probe_cache(target)
    cache[(_normalise_url(url), bare_id)] = frozenset(capabilities)
    _write(cache, target)


def prune_probe_cache(*, known: set[tuple[str, str]], path: Optional[Path] = None) -> None:
    target = _resolve(path)
    cache = load_probe_cache(target)
    kept = {key: value for key, value in cache.items() if key in known}
    if len(kept) != len(cache):
        _write(kept, target)
