"""Diagnosi dei provider e dei modelli.

Modulo puro: riceve il risultato della scoperta e restituisce un report.
L'unico I/O possibile è il probe, che arriva iniettato come callable così
i test non toccano la rete e la UI decide se pagarne il costo.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, TYPE_CHECKING

from .capabilities import CAP_VISION, SOURCE_HEURISTIC, SOURCE_PROBED
from .model_catalog import catalog_for_role
from .model_selection import ROLE_FACTS, ROLE_VISION, rank_models
from .providers import KIND_OLLAMA, PROVIDERS, split_model_id

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import DiscoveryResult, ModelInfo
    from .settings import Settings

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"

_SEVERITY = {STATUS_SKIP: 0, STATUS_OK: 1, STATUS_WARN: 2, STATUS_FAIL: 3}


@dataclass(frozen=True)
class Remedy:
    kind: str          # "pull" | "hint"
    model: str
    provider: str
    size_bytes: int = 0
    note: str = ""


@dataclass(frozen=True)
class Check:
    key: str
    label: str
    status: str
    detail: str = ""
    remedies: tuple[Remedy, ...] = ()


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[Check, ...] = ()

    @property
    def worst(self) -> str:
        if not self.checks:
            return STATUS_SKIP
        return max((c.status for c in self.checks), key=lambda s: _SEVERITY.get(s, 0))

    @property
    def exit_code(self) -> int:
        return 1 if self.worst == STATUS_FAIL else 0


def _provider_check(status) -> Check:
    label = f"{status.name} {status.url}".strip()
    if not status.configured:
        return Check(f"provider.{status.name}", label, STATUS_SKIP, "non configurato")
    if not status.available:
        return Check(f"provider.{status.name}", label, STATUS_FAIL, status.detail)
    if not status.models:
        return Check(f"provider.{status.name}", label, STATUS_WARN,
                     "raggiungibile, nessun modello")
    return Check(f"provider.{status.name}", label, STATUS_OK,
                 f"{len(status.models)} modelli")


def _installable_provider(discovery) -> Optional[str]:
    for spec in PROVIDERS:
        if not spec.supports_install:
            continue
        status = discovery.status(spec.name)
        if status is not None and status.available:
            return spec.name
    return None


def _remedies(discovery, role: str) -> tuple[Remedy, ...]:
    target = _installable_provider(discovery)
    entries = catalog_for_role(role)
    if target:
        return tuple(
            Remedy(kind="pull", model=e.tag, provider=target,
                   size_bytes=e.size_bytes, note=e.note)
            for e in entries
        )
    return tuple(
        Remedy(kind="hint", model=e.tag, provider="", size_bytes=e.size_bytes,
               note="avvia il server con questo modello, oppure configura ollama")
        for e in entries
    )


def _apply_probe(discovery: "DiscoveryResult", probe) -> tuple["ModelInfo", ...]:
    models = discovery.models
    if probe is None:
        return models
    out = []
    for model in models:
        if model.capability_source != SOURCE_HEURISTIC:
            out.append(model)
            continue
        spec, bare = split_model_id(model.id)
        if spec.kind == KIND_OLLAMA:
            out.append(model)
            continue
        status = discovery.status(model.provider)
        if status is None or not status.url:
            out.append(model)
            continue
        try:
            verdict = probe(url=status.url, bare_id=bare)
        except Exception:
            verdict = None
        if verdict is True:
            caps = frozenset(model.capabilities | {CAP_VISION})
            out.append(replace(model, capabilities=caps, capability_source=SOURCE_PROBED))
        elif verdict is False:
            caps = frozenset(model.capabilities - {CAP_VISION})
            out.append(replace(model, capabilities=caps, capability_source=SOURCE_PROBED))
        else:
            out.append(model)
    return tuple(out)


def _role_check(models, *, key: str, label: str, role: str, discovery, pinned: str) -> Check:
    ranked = rank_models(models, role)
    if not ranked:
        return Check(key, label, STATUS_FAIL, "nessun modello disponibile",
                     _remedies(discovery, role))
    if pinned and pinned != "auto" and pinned not in ranked:
        return Check(key, label, STATUS_WARN,
                     f"modello fissato non trovato: {pinned} (userei {ranked[0]})")
    chosen = pinned if (pinned and pinned in ranked) else ranked[0]
    by_id = {m.id: m for m in models}
    if by_id[chosen].capability_source == SOURCE_HEURISTIC:
        return Check(key, label, STATUS_WARN,
                     f"{chosen} (capability dedotta dal nome, non confermata)")
    return Check(key, label, STATUS_OK, chosen)


def run_doctor(
    *,
    discovery: "DiscoveryResult",
    settings: "Settings",
    probe: Optional[Callable[..., Optional[bool]]] = None,
) -> DoctorReport:
    checks = [_provider_check(s) for s in discovery.providers]
    models = _apply_probe(discovery, probe)
    checks.append(_role_check(models, key="role.text", label="modello semantico",
                              role=ROLE_FACTS, discovery=discovery,
                              pinned=settings.facts_model))
    checks.append(_role_check(models, key="role.vision", label="modello vision",
                              role=ROLE_VISION, discovery=discovery,
                              pinned=settings.vision_model))
    return DoctorReport(checks=tuple(checks))
