from __future__ import annotations

from pathlib import Path

from archiver.capabilities import CAP_COMPLETION, CAP_VISION, SOURCE_DECLARED
from archiver.discovery import DiscoveryResult, ModelInfo, ProviderStatus
from archiver.settings import Settings
from archiver.task_builders import build_analysis_config
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)


def _model(model_id, provider, size, *, vision=False):
    caps = {CAP_COMPLETION} | ({CAP_VISION} if vision else set())
    return ModelInfo(
        id=model_id, provider=provider, capabilities=frozenset(caps),
        parameter_size_b=size, capability_source=SOURCE_DECLARED,
    )


def _discovery():
    return DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://vllm.invalid", configured=True, available=True,
                       models=(_model("vllm:qwen3.6-27b", "vllm", 27.0, vision=True),)),
        ProviderStatus(name="ollama", url="http://ollama.invalid", configured=True, available=True,
                       models=(_model("ollama:qwen3:8b", "ollama", 8.2),)),
        ProviderStatus(name="ds4", configured=False),
    ))


def _settings(**kw):
    base = dict(source_root=Path("/tmp/src"), archive_root=Path("/tmp/arc"))
    base.update(kw)
    return Settings(**base)


def test_provider_urls_are_carried_into_the_analysis_config():
    urls = {"ollama": "http://ollama.invalid", "vllm": "http://vllm.invalid", "ds4": ""}
    cfg = build_analysis_config(settings=_settings(providers=urls), discovery=_discovery(), taxonomy=_TAXONOMY)
    assert cfg.provider_urls == urls


def test_auto_selection_prefers_vllm_for_every_role():
    cfg = build_analysis_config(settings=_settings(), discovery=_discovery(), taxonomy=_TAXONOMY)
    assert cfg.text_models[0] == "vllm:qwen3.6-27b"
    assert cfg.vision_models[0] == "vllm:qwen3.6-27b"


def test_a_pinned_model_goes_first_without_dropping_the_others():
    cfg = build_analysis_config(
        settings=_settings(facts_model="ollama:qwen3:8b"),
        discovery=_discovery(),
        taxonomy=_TAXONOMY,
    )
    assert cfg.text_models[0] == "ollama:qwen3:8b"
    assert "vllm:qwen3.6-27b" in cfg.text_models


def test_no_discovery_yields_empty_candidate_lists():
    cfg = build_analysis_config(settings=_settings(), discovery=None, taxonomy=_TAXONOMY)
    assert cfg.text_models == ()
    assert cfg.vision_models == ()
