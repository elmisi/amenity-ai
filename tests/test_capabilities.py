from __future__ import annotations

from archiver.capabilities import (
    CAP_COMPLETION,
    CAP_VISION,
    guess_capabilities,
    interpret_probe,
    parse_parameter_size,
)


def test_size_from_vllm_id_ignores_the_version_number():
    # "qwen3.6-27b": il 3.6 è la versione, il 27 è la taglia.
    assert parse_parameter_size("qwen3.6-27b") == 27.0


def test_size_from_vllm_root_path():
    assert parse_parameter_size("/models/Qwen3.6-27B-AWQ-INT4") == 27.0


def test_size_from_ollama_parameter_size_field():
    assert parse_parameter_size("8.2B") == 8.2


def test_size_from_ollama_tag():
    assert parse_parameter_size("gemma3:1b") == 1.0
    assert parse_parameter_size("llama3.3:70b") == 70.0


def test_size_is_none_when_nothing_says_it():
    assert parse_parameter_size("phi4-mini:latest") is None
    assert parse_parameter_size("") is None


def test_size_takes_first_match_across_arguments_in_order():
    # root ha la precedenza sull'id perché è più informativo
    assert parse_parameter_size("", "/models/Qwen-235B-A22B", "qwen") == 235.0


def test_vision_heuristic_recognises_known_families():
    for name in ("llava:7b", "moondream:latest", "minicpm-v:latest",
                 "bakllava:latest", "Qwen2.5-VL-7B-Instruct", "pixtral-12b"):
        assert CAP_VISION in guess_capabilities(model_id=name), name


def test_vision_heuristic_misses_qwen36_27b_which_is_why_the_probe_exists():
    # Falso negativo verificato sul campo il 2026-08-01: il modello ACCETTA
    # immagini ma il nome non lo dice. Solo il probe può correggerlo.
    caps = guess_capabilities(model_id="qwen3.6-27b")
    assert caps == frozenset({CAP_COMPLETION})


def test_probe_200_confirms_vision():
    assert interpret_probe(status=200, body='{"choices":[]}') is True


def test_probe_400_about_images_confirms_text_only():
    body = '{"error":{"message":"This model does not support image input"}}'
    assert interpret_probe(status=400, body=body) is False


def test_probe_is_inconclusive_on_timeout_or_server_error():
    assert interpret_probe(status=500, body="boom") is None
    assert interpret_probe(status=0, body="") is None


def test_probe_is_inconclusive_on_400_unrelated_to_images():
    body = '{"error":{"message":"max_tokens too large"}}'
    assert interpret_probe(status=400, body=body) is None
