import threading
import time

import pytest

from archiver.concurrency import (
    MAX_SLOTS,
    ConcurrencyLimiter,
    clamp_limit,
    pool_size_for,
)


def test_clamp_rejects_nonsense_and_keeps_the_default():
    assert clamp_limit("", default=4) == 4
    assert clamp_limit("abc", default=4) == 4
    assert clamp_limit(None, default=3) == 3


def test_clamp_pins_the_range():
    # 0 is legal since 0.16.0: it disables the provider.
    assert clamp_limit(0, default=4) == 0
    assert clamp_limit(-7, default=4) == 0
    assert clamp_limit(999, default=4) == MAX_SLOTS
    assert clamp_limit("6", default=4) == 6


def test_missing_providers_fall_back_to_the_registry_default():
    limiter = ConcurrencyLimiter.from_limits({"vllm": 2})
    assert limiter.limit("vllm") == 2
    assert limiter.limit("ds4") == 1


def test_unknown_provider_names_are_ignored():
    limiter = ConcurrencyLimiter.from_limits({"nope": 8})
    assert "nope" not in limiter.limits


def test_an_unknown_provider_is_not_limited():
    limiter = ConcurrencyLimiter.from_limits({})
    with limiter.slot("nope"):
        pass  # must not block, must not raise


def test_the_third_caller_waits_until_a_slot_frees():
    limiter = ConcurrencyLimiter.from_limits({"vllm": 2})
    entered = threading.Semaphore(0)
    release = threading.Event()
    inside: list[int] = []

    def worker() -> None:
        with limiter.slot("vllm"):
            inside.append(1)
            entered.release()
            release.wait(timeout=5)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    assert entered.acquire(timeout=5)
    assert entered.acquire(timeout=5)
    time.sleep(0.05)
    assert len(inside) == 2, "the third caller must still be waiting"
    release.set()
    for t in threads:
        t.join(timeout=5)
    assert len(inside) == 3


def test_the_slot_is_released_when_the_call_raises():
    limiter = ConcurrencyLimiter.from_limits({"vllm": 1})
    with pytest.raises(RuntimeError):
        with limiter.slot("vllm"):
            raise RuntimeError("boom")
    acquired = limiter.slots["vllm"].acquire(timeout=1)
    assert acquired, "a raising call must not leak its slot"


def test_pool_is_sized_on_the_configured_providers():
    limiter = ConcurrencyLimiter.from_limits({"vllm": 4, "ollama": 1})
    assert pool_size_for({"vllm": "http://example.invalid", "ollama": ""}, limiter) == 4
    assert pool_size_for({"vllm": "", "ollama": "http://example.invalid"}, limiter) == 1
    assert pool_size_for({"vllm": "", "ollama": "", "ds4": ""}, limiter) == 1
