from __future__ import annotations

from pathlib import Path

from archiver.capabilities import (
    CAP_COMPLETION,
    CAP_VISION,
    SOURCE_DECLARED,
    SOURCE_HEURISTIC,
)
from archiver.discovery import DiscoveryResult, ModelInfo, ProviderStatus
from archiver.doctor import run_doctor
from archiver.settings import Settings


def _settings(**kw):
    base = dict(source_root=Path("/tmp/src"), archive_root=Path("/tmp/arc"))
    base.update(kw)
    return Settings(**base)


def _model(model_id, provider, *, vision=False, source=SOURCE_DECLARED, size=8.0):
    caps = {CAP_COMPLETION} | ({CAP_VISION} if vision else set())
    return ModelInfo(id=model_id, provider=provider, capabilities=frozenset(caps),
                     parameter_size_b=size, capability_source=source)


def _check(report, key):
    return next(c for c in report.checks if c.key == key)


def test_report_has_one_check_per_provider_plus_two_roles():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", configured=False),
        ProviderStatus(name="ollama", configured=False),
        ProviderStatus(name="ds4", configured=False),
    )), settings=_settings())
    keys = [c.key for c in report.checks]
    assert keys == ["provider.vllm", "provider.ollama", "provider.ds4",
                    "role.text", "role.vision"]


def test_unconfigured_provider_is_skipped_not_failed():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", configured=False),
    )), settings=_settings())
    assert _check(report, "provider.vllm").status == "skip"


def test_unreachable_provider_fails_with_the_real_reason():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://x.invalid", configured=True,
                       available=False, detail="ConnectionRefusedError: nope"),
    )), settings=_settings())
    check = _check(report, "provider.vllm")
    assert check.status == "fail"
    assert "ConnectionRefusedError" in check.detail


def test_reachable_provider_without_models_warns():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="ollama", url="http://x.invalid", configured=True,
                       available=True, models=()),
    )), settings=_settings())
    assert _check(report, "provider.ollama").status == "warn"


def test_missing_vision_fails_and_offers_a_pull_when_ollama_is_reachable():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="ollama", url="http://x.invalid", configured=True,
                       available=True, models=(_model("ollama:qwen3:8b", "ollama"),)),
    )), settings=_settings())
    vision = _check(report, "role.vision")
    assert vision.status == "fail"
    assert vision.remedies
    assert all(r.kind == "pull" and r.provider == "ollama" for r in vision.remedies)


def test_missing_vision_offers_only_hints_when_no_installable_provider_is_up():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://x.invalid", configured=True,
                       available=True, models=(_model("vllm:qwen3.6-27b", "vllm"),)),
        ProviderStatus(name="ollama", configured=False),
    )), settings=_settings())
    vision = _check(report, "role.vision")
    assert vision.status == "fail"
    assert all(r.kind == "hint" for r in vision.remedies)


def test_heuristic_only_vision_warns_instead_of_passing():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://x.invalid", configured=True, available=True,
                       models=(_model("vllm:llava-ish", "vllm", vision=True,
                                      source=SOURCE_HEURISTIC),)),
    )), settings=_settings())
    assert _check(report, "role.vision").status == "warn"


def test_probe_promotes_a_heuristic_vision_model_to_ok():
    model = _model("vllm:qwen3.6-27b", "vllm", source=SOURCE_HEURISTIC)
    discovery = DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://x.invalid", configured=True,
                       available=True, models=(model,)),
    ))

    def probe(*, url, bare_id):
        return True  # il modello accetta immagini

    report = run_doctor(discovery=discovery, settings=_settings(), probe=probe)
    assert _check(report, "role.vision").status == "ok"


def test_probe_is_not_called_for_declared_capabilities():
    calls = []

    def probe(*, url, bare_id):
        calls.append(bare_id)
        return True

    run_doctor(
        discovery=DiscoveryResult(providers=(
            ProviderStatus(name="ollama", url="http://x.invalid", configured=True, available=True,
                           models=(_model("ollama:llava:7b", "ollama", vision=True),)),
        )),
        settings=_settings(),
        probe=probe,
    )
    assert calls == []


def test_pinned_model_that_disappeared_warns():
    report = run_doctor(
        discovery=DiscoveryResult(providers=(
            ProviderStatus(name="ollama", url="http://x.invalid", configured=True, available=True,
                           models=(_model("ollama:qwen3:8b", "ollama"),)),
        )),
        settings=_settings(facts_model="ollama:disappeared"),
    )
    text = _check(report, "role.text")
    assert text.status == "warn"
    assert "disappeared" in text.detail


def test_worst_and_exit_code_reflect_the_severity():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="ollama", configured=False),
    )), settings=_settings())
    assert report.worst == "fail"   # nessun modello di testo
    assert report.exit_code == 1
