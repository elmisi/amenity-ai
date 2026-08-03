"""Per-provider request limits.

vLLM batches concurrent requests and stops improving past four; ds4 answers one
caller at a time. The limit therefore belongs to the provider, and it has to be
applied where the provider is known — at the call site in the router — because
analysis walks a candidate list and two files of the same run can land on
different providers.

Nothing here is a module-level singleton: the limiter is built per run and
passed down beside provider_urls, so tests stay independent of each other.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

from .providers import default_provider_concurrency

MAX_SLOTS = 16


def clamp_limit(value: object, *, default: int) -> int:
    """Coerce user input to a usable slot count, falling back to `default`."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(1, min(MAX_SLOTS, n))


@dataclass(frozen=True)
class ConcurrencyLimiter:
    slots: Mapping[str, threading.Semaphore]
    limits: Mapping[str, int]

    @classmethod
    def from_limits(cls, limits: Mapping[str, int] | None = None) -> "ConcurrencyLimiter":
        merged = default_provider_concurrency()
        for name, value in (limits or {}).items():
            if name in merged:
                merged[name] = clamp_limit(value, default=merged[name])
        return cls(
            slots={name: threading.Semaphore(n) for name, n in merged.items()},
            limits=dict(merged),
        )

    def limit(self, provider: str) -> int:
        return self.limits.get(provider, 1)

    @contextmanager
    def slot(self, provider: str) -> Iterator[None]:
        """Hold a slot for `provider`. Unknown names are not limited."""
        sem = self.slots.get(provider)
        if sem is None:
            yield
            return
        sem.acquire()
        try:
            yield
        finally:
            sem.release()


def pool_size_for(provider_urls: Mapping[str, str], limiter: ConcurrencyLimiter) -> int:
    """Worker count for a run: the widest limit among configured providers.

    The semaphores are the real regulator — a thread that cannot get a slot
    waits while its peers extract text — so the pool only has to be wide enough
    not to become the bottleneck itself.
    """
    configured = [name for name, url in (provider_urls or {}).items() if (url or "").strip()]
    if not configured:
        return 1
    return max(1, min(MAX_SLOTS, max(limiter.limit(name) for name in configured)))
