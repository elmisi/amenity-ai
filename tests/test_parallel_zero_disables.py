"""parallel: 0 disables a provider while its URL stays in the config.

Blanking the URL was the only off switch, and it forgets the endpoint. Zero
slots means "never call it": discovery skips the provider, the doctor reports
it as disabled instead of failed, and no model of its enters the ranking.
"""
from pathlib import Path

import pytest

from archiver.concurrency import ConcurrencyLimiter, clamp_limit
from archiver.discovery import discover_providers
from archiver.doctor import STATUS_SKIP, run_doctor
from archiver.settings import Settings


def _settings(**kw) -> Settings:
    base = dict(
        source_root=Path("/tmp/s"),
        archive_root=Path("/tmp/a"),
        providers={"vllm": "http://example.invalid:8000", "ollama": "http://example.invalid:11434", "ds4": ""},
    )
    base.update(kw)
    return Settings(**base)


def test_zero_is_a_legal_limit_now():
    assert clamp_limit(0, default=4) == 0
    assert clamp_limit("0", default=4) == 0


def test_nonsense_still_falls_back_to_the_default_not_to_zero():
    assert clamp_limit("", default=4) == 4
    assert clamp_limit("abc", default=4) == 4


def test_settings_reports_which_providers_are_off():
    s = _settings(provider_concurrency={"ollama": 0})
    assert s.disabled_providers() == frozenset({"ollama"})


def test_enabled_urls_mask_the_disabled_provider_but_keep_the_stored_one():
    s = _settings(provider_concurrency={"ollama": 0})
    assert s.enabled_provider_urls()["ollama"] == ""
    assert s.providers["ollama"] == "http://example.invalid:11434", "the URL must survive"
    assert s.enabled_provider_urls()["vllm"] == "http://example.invalid:8000"


def test_discovery_never_contacts_a_disabled_provider():
    def fetch(url, *, timeout_s):
        if "11434" in url:
            pytest.fail("the disabled provider was contacted")
        return {"data": [{"id": "m"}]}

    result = discover_providers(
        {"vllm": "http://example.invalid:8000", "ollama": "http://example.invalid:11434", "ds4": ""},
        fetch=fetch,
        disabled={"ollama"},
    )
    status = result.status("ollama")
    assert status.configured is False
    assert status.available is False
    assert status.detail == "disabled (parallel 0)"
    assert status.url == "http://example.invalid:11434", "the doctor line should still show it"


def test_the_doctor_skips_a_disabled_provider_instead_of_failing():
    def fetch(url, *, timeout_s):
        return {"data": [{"id": "qwen-vl-anything"}]}

    discovery = discover_providers(
        {"vllm": "http://example.invalid:8000", "ollama": "http://example.invalid:11434", "ds4": ""},
        fetch=fetch,
        disabled={"ollama"},
    )
    report = run_doctor(discovery=discovery, settings=_settings(), probe=None)
    check = next(c for c in report.checks if c.key == "provider.ollama")
    assert check.status == STATUS_SKIP
    assert check.detail == "disabled (parallel 0)"


def test_a_zero_limit_cannot_deadlock_a_stray_call():
    limiter = ConcurrencyLimiter.from_limits({"vllm": 0})
    assert limiter.limit("vllm") == 0
    with limiter.slot("vllm"):
        pass  # must not block: routing is the gate, the semaphore is not
