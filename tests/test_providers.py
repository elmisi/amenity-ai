from __future__ import annotations

import pytest

from archiver.providers import (
    PROVIDERS,
    PROVIDER_NAMES,
    default_provider_urls,
    join_model_id,
    provider_by_name,
    provider_priority,
    split_model_id,
)


def test_priority_order_is_vllm_then_ollama_then_ds4():
    assert [p.name for p in PROVIDERS] == ["vllm", "ollama", "ds4"]
    assert provider_priority("vllm") < provider_priority("ollama") < provider_priority("ds4")


def test_split_uses_known_prefix_not_first_colon():
    spec, bare = split_model_id("ollama:qwen3:8b")
    assert spec.name == "ollama"
    assert bare == "qwen3:8b"


def test_split_treats_bare_legacy_id_as_ollama():
    # 0.11.0 configs saved "qwen3:8b" with no prefix: it must never
    # produrre un provider inesistente "qwen3".
    spec, bare = split_model_id("qwen3:8b")
    assert spec.name == "ollama"
    assert bare == "qwen3:8b"


def test_split_recognises_openai_compat_prefixes():
    assert split_model_id("vllm:qwen3.6-27b")[0].name == "vllm"
    assert split_model_id("vllm:qwen3.6-27b")[1] == "qwen3.6-27b"
    assert split_model_id("ds4:deepseek-v4-flash")[0].name == "ds4"


@pytest.mark.parametrize("provider,bare", [
    ("ollama", "qwen3:8b"),
    ("vllm", "qwen3.6-27b"),
    ("ds4", "deepseek-v4-flash"),
])
def test_join_split_round_trip(provider, bare):
    joined = join_model_id(provider, bare)
    spec, out = split_model_id(joined)
    assert (spec.name, out) == (provider, bare)


def test_only_ollama_supports_install():
    assert [p.name for p in PROVIDERS if p.supports_install] == ["ollama"]


def test_each_openai_compat_provider_knows_how_to_switch_reasoning_off():
    # Measured on 2026-08-02 against vLLM 0.21 with qwen3.6-27b:
    # reasoning_effort is accepted and ignored (55.2s either way), while
    # enable_thinking=False brings the same request down to 2.2s.
    # One lever per provider, not a single shared one.
    by_name = {p.name: p for p in PROVIDERS}
    assert by_name["vllm"].thinking_off == {"chat_template_kwargs": {"enable_thinking": False}}
    assert by_name["ds4"].thinking_off == {"reasoning_effort": "low"}
    # Ollama ha il suo parametro `think` nativo, gestito dal suo backend.
    assert by_name["ollama"].thinking_off == {}


def test_default_urls_have_no_real_hostnames():
    urls = default_provider_urls()
    assert set(urls) == set(PROVIDER_NAMES)
    assert urls["ollama"] == "http://localhost:11434"
    assert urls["vllm"] == ""
    assert urls["ds4"] == ""


def test_provider_by_name_returns_none_for_unknown():
    assert provider_by_name("nope") is None
