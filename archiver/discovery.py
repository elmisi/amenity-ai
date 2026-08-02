"""Discovery of the models available on the configured providers.

One request per provider, the three of them in parallel: the worst case
stays a single timeout rather than their sum. Ollama >= 0.31 already
declares capabilities and parameter_size in /api/tags, so no per-model
request and no cache of declared capabilities are needed.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional
from urllib.request import urlopen

from .capabilities import (
    SOURCE_DECLARED,
    SOURCE_HEURISTIC,
    SOURCE_PROBED,
    guess_capabilities,
    parse_parameter_size,
)
from .providers import KIND_OLLAMA, PROVIDERS, join_model_id, split_model_id


@dataclass(frozen=True)
class ModelInfo:
    id: str
    provider: str
    capabilities: frozenset[str] = frozenset()
    parameter_size_b: Optional[float] = None
    context_length: Optional[int] = None
    capability_source: str = SOURCE_HEURISTIC


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    url: str = ""
    configured: bool = False
    available: bool = False
    detail: str = ""
    models: tuple[ModelInfo, ...] = ()


@dataclass(frozen=True)
class DiscoveryResult:
    providers: tuple[ProviderStatus, ...] = ()

    @property
    def models(self) -> tuple[ModelInfo, ...]:
        out: list[ModelInfo] = []
        for status in self.providers:
            out.extend(status.models)
        return tuple(out)

    def status(self, name: str) -> Optional[ProviderStatus]:
        for status in self.providers:
            if status.name == name:
                return status
        return None


def _get_json(url: str, *, timeout_s: float) -> Any:
    with urlopen(url, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _as_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and value > 0 else None


def parse_ollama_tags(payload: Any) -> tuple[ModelInfo, ...]:
    entries = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return ()
    out: list[ModelInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        declared = entry.get("capabilities")
        if isinstance(declared, list) and declared:
            caps = frozenset(c for c in declared if isinstance(c, str) and c.strip())
            source = SOURCE_DECLARED
        else:
            caps = guess_capabilities(model_id=name)
            source = SOURCE_HEURISTIC
        out.append(
            ModelInfo(
                id=join_model_id("ollama", name),
                provider="ollama",
                capabilities=caps,
                parameter_size_b=parse_parameter_size(
                    str(details.get("parameter_size") or ""), name
                ),
                context_length=_as_int(details.get("context_length")),
                capability_source=source,
            )
        )
    return tuple(out)


def parse_openai_models(payload: Any, *, provider_name: str) -> tuple[ModelInfo, ...]:
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return ()
    out: list[ModelInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        root = entry.get("root") if isinstance(entry.get("root"), str) else ""
        out.append(
            ModelInfo(
                id=join_model_id(provider_name, model_id),
                provider=provider_name,
                capabilities=guess_capabilities(model_id=model_id, root=root),
                parameter_size_b=parse_parameter_size(root, model_id),
                context_length=_as_int(entry.get("max_model_len")),
                capability_source=SOURCE_HEURISTIC,
            )
        )
    return tuple(out)


def _apply_probe_cache(
    models: tuple[ModelInfo, ...],
    *,
    url: str,
    probe_cache: Mapping[tuple[str, str], frozenset[str]],
) -> tuple[ModelInfo, ...]:
    if not probe_cache:
        return models
    key_url = url.rstrip("/")
    out: list[ModelInfo] = []
    for model in models:
        _, bare = split_model_id(model.id)
        cached = probe_cache.get((key_url, bare))
        if cached:
            out.append(
                replace(model, capabilities=cached, capability_source=SOURCE_PROBED)
            )
        else:
            out.append(model)
    return tuple(out)


def _probe_one(
    spec, url: str, *, fetch: Callable[..., Any], timeout_s: float
) -> ProviderStatus:
    url = url.strip()
    if not url:
        return ProviderStatus(name=spec.name, configured=False, detail="not configured")
    endpoint = url.rstrip("/") + ("/api/tags" if spec.kind == KIND_OLLAMA else "/v1/models")
    try:
        payload = fetch(endpoint, timeout_s=timeout_s)
    except Exception as exc:
        return ProviderStatus(
            name=spec.name,
            url=url,
            configured=True,
            available=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    if spec.kind == KIND_OLLAMA:
        models = parse_ollama_tags(payload)
    else:
        models = parse_openai_models(payload, provider_name=spec.name)
    return ProviderStatus(
        name=spec.name,
        url=url,
        configured=True,
        available=True,
        detail="ok" if models else "reachable, no models",
        models=models,
    )


def discover_providers(
    provider_urls: Mapping[str, str],
    *,
    fetch: Optional[Callable[..., Any]] = None,
    probe_cache: Optional[Mapping[tuple[str, str], frozenset[str]]] = None,
    timeout_s: float = 2.5,
) -> DiscoveryResult:
    fetcher = fetch or _get_json
    cache = probe_cache if probe_cache is not None else {}

    def work(spec) -> ProviderStatus:
        status = _probe_one(
            spec, provider_urls.get(spec.name, "") or "", fetch=fetcher, timeout_s=timeout_s
        )
        if status.models and cache:
            status = replace(
                status, models=_apply_probe_cache(status.models, url=status.url, probe_cache=cache)
            )
        return status

    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as pool:
        statuses = tuple(pool.map(work, PROVIDERS))
    return DiscoveryResult(providers=statuses)
