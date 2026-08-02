from __future__ import annotations

from archiver.capabilities import (
    CAP_COMPLETION,
    CAP_EMBEDDING,
    CAP_VISION,
    SOURCE_DECLARED,
    SOURCE_HEURISTIC,
    SOURCE_PROBED,
)
from archiver.discovery import (
    discover_providers,
    parse_ollama_tags,
    parse_openai_models,
)

# Payload reale catturato da Ollama 0.31.1 il 2026-08-01.
OLLAMA_TAGS = {
    "models": [
        {
            "name": "qwen3:8b",
            "model": "qwen3:8b",
            "modified_at": "2026-07-03T10:07:48.059344839+02:00",
            "size": 5225388164,
            "digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
            "details": {
                "family": "qwen3",
                "parameter_size": "8.2B",
                "quantization_level": "Q4_K_M",
                "context_length": 40960,
            },
            "capabilities": ["completion", "tools", "thinking"],
        }
    ]
}

# Payload reale catturato da vLLM 0.21.0 il 2026-08-01.
VLLM_MODELS = {
    "object": "list",
    "data": [
        {
            "id": "qwen3.6-27b",
            "object": "model",
            "owned_by": "vllm",
            "root": "/models/Qwen3.6-27B-AWQ-INT4",
            "max_model_len": 131072,
        }
    ],
}


def test_ollama_tags_are_parsed_as_declared_capabilities():
    models = parse_ollama_tags(OLLAMA_TAGS)
    assert len(models) == 1
    m = models[0]
    assert m.id == "ollama:qwen3:8b"
    assert m.provider == "ollama"
    assert m.capability_source == SOURCE_DECLARED
    assert CAP_COMPLETION in m.capabilities
    assert CAP_VISION not in m.capabilities
    assert m.parameter_size_b == 8.2
    assert m.context_length == 40960


def test_ollama_entry_without_capabilities_falls_back_to_heuristic():
    payload = {"models": [{"name": "llava:7b", "details": {"parameter_size": "7B"}}]}
    m = parse_ollama_tags(payload)[0]
    assert m.capability_source == SOURCE_HEURISTIC
    assert CAP_VISION in m.capabilities


def test_openai_models_are_parsed_as_heuristic_with_size_from_root():
    models = parse_openai_models(VLLM_MODELS, provider_name="vllm")
    assert len(models) == 1
    m = models[0]
    assert m.id == "vllm:qwen3.6-27b"
    assert m.provider == "vllm"
    assert m.capability_source == SOURCE_HEURISTIC
    # The name heuristic does NOT see multimodality: only the probe corrects it.
    assert CAP_VISION not in m.capabilities
    assert m.parameter_size_b == 27.0
    assert m.context_length == 131072


def test_non_list_payloads_are_rejected_without_raising():
    assert parse_ollama_tags({"models": "nope"}) == ()
    assert parse_ollama_tags("nope") == ()
    assert parse_openai_models({"data": {"id": "x"}}, provider_name="vllm") == ()
    assert parse_openai_models(None, provider_name="vllm") == ()


def test_embedding_models_keep_their_declared_capability():
    payload = {"models": [{"name": "nomic-embed-text:latest",
                           "capabilities": ["embedding"]}]}
    m = parse_ollama_tags(payload)[0]
    assert m.capabilities == frozenset({CAP_EMBEDDING})


def _fetch_from(mapping):
    def fetch(url, *, timeout_s):
        for fragment, payload in mapping.items():
            if fragment in url:
                return payload
        raise ConnectionRefusedError("nothing here")
    return fetch


def test_empty_url_means_not_configured_and_no_request():
    calls = []

    def fetch(url, *, timeout_s):
        calls.append(url)
        return OLLAMA_TAGS

    result = discover_providers(
        {"ollama": "http://ollama.invalid", "vllm": "", "ds4": ""},
        fetch=fetch,
    )
    vllm = result.status("vllm")
    assert vllm.configured is False
    assert vllm.available is False
    assert vllm.models == ()
    assert all("vllm" not in url for url in calls)


def test_unreachable_provider_reports_the_real_reason():
    result = discover_providers(
        {"ollama": "http://ollama.invalid", "vllm": "http://vllm.invalid", "ds4": ""},
        fetch=_fetch_from({"/api/tags": OLLAMA_TAGS}),
    )
    vllm = result.status("vllm")
    assert vllm.configured is True
    assert vllm.available is False
    assert "ConnectionRefusedError" in vllm.detail


def test_models_property_merges_every_available_provider():
    result = discover_providers(
        {"ollama": "http://ollama.invalid", "vllm": "http://vllm.invalid", "ds4": ""},
        fetch=_fetch_from({"/api/tags": OLLAMA_TAGS, "/v1/models": VLLM_MODELS}),
    )
    assert {m.id for m in result.models} == {"ollama:qwen3:8b", "vllm:qwen3.6-27b"}


def test_probe_cache_promotes_capabilities_to_probed():
    cache = {("http://vllm.invalid", "qwen3.6-27b"): frozenset({CAP_COMPLETION, CAP_VISION})}
    result = discover_providers(
        {"ollama": "", "vllm": "http://vllm.invalid", "ds4": ""},
        fetch=_fetch_from({"/v1/models": VLLM_MODELS}),
        probe_cache=cache,
    )
    m = result.status("vllm").models[0]
    assert m.capability_source == SOURCE_PROBED
    assert CAP_VISION in m.capabilities


def test_reachable_provider_with_zero_models_is_available_but_empty():
    result = discover_providers(
        {"ollama": "http://ollama.invalid", "vllm": "", "ds4": ""},
        fetch=_fetch_from({"/api/tags": {"models": []}}),
    )
    ollama = result.status("ollama")
    assert ollama.available is True
    assert ollama.models == ()
