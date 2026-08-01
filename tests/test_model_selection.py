from __future__ import annotations

from archiver.capabilities import CAP_COMPLETION, CAP_EMBEDDING, CAP_VISION, SOURCE_DECLARED
from archiver.discovery import ModelInfo
from archiver.model_selection import (
    ROLE_CLASSIFY,
    ROLE_FACTS,
    ROLE_VISION,
    rank_models,
    size_bucket,
)


def m(model_id, provider, *, size=None, vision=False, embedding=False):
    caps = {CAP_EMBEDDING} if embedding else {CAP_COMPLETION}
    if vision:
        caps.add(CAP_VISION)
    return ModelInfo(
        id=model_id,
        provider=provider,
        capabilities=frozenset(caps),
        parameter_size_b=size,
        capability_source=SOURCE_DECLARED,
    )


def test_size_buckets_have_five_levels():
    assert size_bucket(1.0) == 0
    assert size_bucket(3.0) == 1
    assert size_bucket(8.2) == 2
    assert size_bucket(13.0) == 3
    assert size_bucket(232.0) == 4
    assert size_bucket(None) is None


def test_provider_priority_comes_before_size():
    models = [
        m("ollama:qwen3:8b", "ollama", size=8.2),
        m("vllm:qwen3.6-27b", "vllm", size=27.0),
        m("ds4:deepseek-v4-flash", "ds4", size=232.0),
    ]
    assert rank_models(models, ROLE_FACTS)[0] == "vllm:qwen3.6-27b"
    assert rank_models(models, ROLE_CLASSIFY)[0] == "vllm:qwen3.6-27b"


def test_ollama_wins_when_vllm_is_absent():
    models = [
        m("ollama:qwen3:8b", "ollama", size=8.2),
        m("ds4:deepseek-v4-flash", "ds4", size=232.0),
    ]
    assert rank_models(models, ROLE_FACTS)[0] == "ollama:qwen3:8b"


def test_within_one_provider_facts_prefers_the_smaller_bucket():
    models = [
        m("ollama:llama3.3:70b", "ollama", size=70.0),
        m("ollama:gemma3:1b", "ollama", size=1.0),
        m("ollama:qwen3:8b", "ollama", size=8.2),
    ]
    assert rank_models(models, ROLE_FACTS) == (
        "ollama:gemma3:1b",
        "ollama:qwen3:8b",
        "ollama:llama3.3:70b",
    )


def test_within_one_provider_classify_prefers_the_5_to_9b_bucket():
    models = [
        m("ollama:gemma3:1b", "ollama", size=1.0),
        m("ollama:qwen3:8b", "ollama", size=8.2),
        m("ollama:llama3.3:70b", "ollama", size=70.0),
    ]
    assert rank_models(models, ROLE_CLASSIFY)[0] == "ollama:qwen3:8b"


def test_curated_bias_breaks_ties_inside_a_bucket():
    # Stessa fascia (2-5B) e stesso provider: decide la lista curata,
    # dove qwen2.5:3b-instruct precede gemma2:2b.
    models = [
        m("ollama:gemma2:2b", "ollama", size=2.0),
        m("ollama:qwen2.5:3b-instruct", "ollama", size=3.0),
    ]
    assert rank_models(models, ROLE_FACTS)[0] == "ollama:qwen2.5:3b-instruct"


def test_curated_bias_matches_on_the_bare_id():
    # Lo stesso modello servito da vLLM deve beneficiare della voce curata
    # scritta senza prefisso.
    models = [
        m("vllm:unknown-3b", "vllm", size=3.0),
        m("vllm:qwen2.5:3b-instruct", "vllm", size=3.0),
    ]
    assert rank_models(models, ROLE_FACTS)[0] == "vllm:qwen2.5:3b-instruct"


def test_unknown_size_goes_last_not_assumed_small():
    models = [
        m("ollama:mystery", "ollama", size=None),
        m("ollama:qwen3:8b", "ollama", size=8.2),
    ]
    assert rank_models(models, ROLE_FACTS)[-1] == "ollama:mystery"


def test_embedding_models_are_excluded_from_text_roles():
    models = [
        m("ollama:nomic-embed-text", "ollama", size=0.1, embedding=True),
        m("ollama:qwen3:8b", "ollama", size=8.2),
    ]
    assert rank_models(models, ROLE_FACTS) == ("ollama:qwen3:8b",)


def test_vision_role_only_returns_models_with_the_capability():
    models = [
        m("ollama:qwen3:8b", "ollama", size=8.2),
        m("vllm:qwen3.6-27b", "vllm", size=27.0, vision=True),
    ]
    assert rank_models(models, ROLE_VISION) == ("vllm:qwen3.6-27b",)


def test_ordering_is_deterministic_for_fully_tied_models():
    models = [
        m("ollama:zzz:3b", "ollama", size=3.0),
        m("ollama:aaa:3b", "ollama", size=3.0),
    ]
    assert rank_models(models, ROLE_FACTS) == ("ollama:aaa:3b", "ollama:zzz:3b")


def test_empty_input_yields_empty_output():
    assert rank_models([], ROLE_FACTS) == ()
