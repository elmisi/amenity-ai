from archiver.providers import (
    PROVIDER_NAMES,
    default_provider_concurrency,
    provider_by_name,
)


def test_vllm_allows_four_concurrent_requests():
    """Measured knee of the real server: past four, throughput stops improving."""
    assert provider_by_name("vllm").max_concurrency == 4


def test_ds4_is_serialised_because_it_is_mutually_exclusive():
    assert provider_by_name("ds4").max_concurrency == 1


def test_ollama_defaults_to_one_since_num_parallel_is_invisible_to_us():
    assert provider_by_name("ollama").max_concurrency == 1


def test_default_concurrency_covers_every_provider():
    assert set(default_provider_concurrency()) == set(PROVIDER_NAMES)
    assert all(v >= 1 for v in default_provider_concurrency().values())
