from pathlib import Path

from archiver import analyzer
from archiver.concurrency import ConcurrencyLimiter
from archiver.ollama_client import OllamaGenerateResult
from archiver.settings import Settings
from archiver.task_builders import build_analysis_config
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines


def test_build_analysis_config_carries_the_limiter():
    taxonomy, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)
    limiter = ConcurrencyLimiter.from_limits({"vllm": 2})
    cfg = build_analysis_config(
        settings=Settings(source_root=Path("/tmp/s"), archive_root=Path("/tmp/a")),
        discovery=None,
        taxonomy=taxonomy,
        limiter=limiter,
    )
    assert cfg.limiter is limiter


def test_the_config_defaults_to_no_limiter():
    taxonomy, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)
    cfg = build_analysis_config(
        settings=Settings(source_root=Path("/tmp/s"), archive_root=Path("/tmp/a")),
        discovery=None,
        taxonomy=taxonomy,
    )
    assert cfg.limiter is None


def test_the_facts_call_forwards_the_limiter_to_the_router(monkeypatch):
    seen: dict[str, object] = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return OllamaGenerateResult(
            response='{"summary_long": "x", "doc_type": "note"}', model="m", done=True
        )

    monkeypatch.setattr(analyzer, "generate", fake_generate)
    limiter = ConcurrencyLimiter.from_limits({"vllm": 2})
    analyzer._extract_facts_from_text(
        model="vllm:m",
        content="text",
        filename="f.pdf",
        mtime_iso="2026-01-01T00:00:00",
        provider_urls={"vllm": "http://example.invalid"},
        year_hint_filename=None,
        year_hint_text=None,
        output_language="en",
        limiter=limiter,
    )
    assert seen["limiter"] is limiter


def test_the_vision_call_forwards_the_limiter(monkeypatch, tmp_path):
    from archiver.extractors import image as image_extractor

    seen: dict[str, object] = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return OllamaGenerateResult(response="a photo of a cat", model="v", done=True)

    monkeypatch.setattr(image_extractor, "generate_with_image_file", fake)
    limiter = ConcurrencyLimiter.from_limits({"vllm": 2})
    picture = tmp_path / "p.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n")
    image_extractor.caption_image(
        picture,
        vision_models=("vllm:v",),
        prompt="describe",
        provider_urls={"vllm": "http://example.invalid"},
        limiter=limiter,
    )
    assert seen["limiter"] is limiter
