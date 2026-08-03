# Parallel Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** run several files through the facts phase at once, bounded per provider, with a TUI that reports what is in flight, what is queued and how long is left.

**Architecture:** a `ConcurrencyLimiter` holding one semaphore per provider is built from config and passed down beside `provider_urls`; `llm_router.generate` acquires a slot around the backend call. The facts worker becomes a `ThreadPoolExecutor`; the semaphores, not the pool size, are the real regulator. Progress is tracked per run and rendered by pure functions.

**Tech Stack:** Python 3.10+, `concurrent.futures`, `threading.Semaphore`, Textual, pytest.

Design doc: `docs/superpowers/specs/2026-08-03-parallel-scan-design.md`.

## Global Constraints

- **No endpoint hostname may appear in any tracked file.** Defaults stay `http://localhost:11434` for Ollama and `""` for vLLM/ds4. Tests use reserved domains only (`example.invalid`).
- **English everywhere** — comments, docstrings, UI strings, commit messages, PR body. Italian stays only in linguistic *data* (`utils_parsing.py` stopwords/months and their test fixtures).
- **No `Co-Authored-By: Claude`**, and no "Generated with Claude Code" footer, in any commit or document.
- **Behaviour-preserving where not explicitly changed**: prompts, heuristics, defaults and UX flows stay as they are.
- **Never block the Textual event loop** — I/O, OCR and LLM calls run in workers; UI updates go through `call_from_thread`.
- Frozen dataclasses for config, results and state. `pathlib.Path` for paths. Type hints on public functions.
- Run tests with `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`.
- Version bump at the end of phase 1: **0.13.0** in both `VERSION` and `pyproject.toml` (`python3 scripts/bump_version.py` bumps patch only — this one is a minor, so edit both by hand and keep them in sync).
- Any TUI verification runs under a PTY: `script -qec "<command>" /dev/null`, with `XDG_CONFIG_HOME` pointed at a scratch dir and `--source` at a scratch folder, so the user's real config and cache are never touched.

---

# Phase 1 — parallel facts (PR 1)

### Task 1: Per-provider concurrency in the registry

**Files:**
- Modify: `archiver/providers.py`
- Test: `tests/test_providers_concurrency.py`

**Interfaces:**
- Produces: `ProviderSpec.max_concurrency: int`, `default_provider_concurrency() -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_concurrency.py
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
```

- [ ] **Step 2: Run it and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_providers_concurrency.py -v`
Expected: FAIL — `ProviderSpec` has no attribute `max_concurrency`.

- [ ] **Step 3: Add the field and the defaults**

In `archiver/providers.py`, add to `ProviderSpec` after `thinking_off`:

```python
    # How many requests this provider answers usefully at once. vLLM batches
    # them; ds4 serves one caller at a time; for Ollama we cannot see
    # OLLAMA_NUM_PARALLEL, and asking for more than it grants only queues
    # server-side while the UI claims work that is not happening.
    max_concurrency: int = 1
```

Then set `max_concurrency=4` on the `vllm` entry (leave `ollama` and `ds4` at the default), and add at the end of the module:

```python
def default_provider_concurrency() -> dict[str, int]:
    return {spec.name: spec.max_concurrency for spec in PROVIDERS}
```

Update the module docstring: `vllm holds up under contention and will scale once scanning runs in parallel` becomes `vllm holds up under contention and is the one that scales when scanning runs in parallel`.

- [ ] **Step 4: Run the tests**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: PASS, whole suite green.

- [ ] **Step 5: Commit**

```bash
git add archiver/providers.py tests/test_providers_concurrency.py
git commit -m "feat: declare a per-provider concurrency limit in the registry"
```

---

### Task 2: The limiter

**Files:**
- Create: `archiver/concurrency.py`
- Test: `tests/test_concurrency.py`

**Interfaces:**
- Consumes: `default_provider_concurrency()` from Task 1
- Produces: `ConcurrencyLimiter.from_limits(mapping) -> ConcurrencyLimiter`, `.slot(provider) -> ContextManager[None]`, `.limit(provider) -> int`, `clamp_limit(value, *, default) -> int`, `pool_size_for(provider_urls, limiter) -> int`, `MAX_SLOTS = 16`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_concurrency.py
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
    assert clamp_limit(0, default=4) == 1
    assert clamp_limit(-7, default=4) == 1
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
```

- [ ] **Step 2: Run it and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_concurrency.py -v`
Expected: FAIL — no module named `archiver.concurrency`.

- [ ] **Step 3: Write the module**

```python
# archiver/concurrency.py
"""Per-provider request limits.

vLLM batches concurrent requests and stops improving past four; ds4 answers one
caller at a time. The limit therefore belongs to the provider, and it has to be
applied where the provider is known — at the call site in the router — because
analysis walks a candidate list and two files of the same run can land on
different providers.

Nothing here is a module-level singleton: the limiter is built per run and
passed down beside provider_urls, so tests are independent of each other.
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
    not to be the bottleneck itself.
    """
    configured = [name for name, url in (provider_urls or {}).items() if (url or "").strip()]
    if not configured:
        return 1
    return max(1, min(MAX_SLOTS, max(limiter.limit(name) for name in configured)))
```

- [ ] **Step 4: Run the tests**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_concurrency.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add archiver/concurrency.py tests/test_concurrency.py
git commit -m "feat: add a per-provider concurrency limiter"
```

---

### Task 3: The router acquires a slot

**Files:**
- Modify: `archiver/llm_router.py`
- Test: `tests/test_llm_router_limiter.py`

**Interfaces:**
- Consumes: `ConcurrencyLimiter` from Task 2
- Produces: `generate(..., limiter=None)` and `generate_with_image_file(..., limiter=None)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_router_limiter.py
from contextlib import contextmanager

from archiver import llm_router
from archiver.llm_backend import LLMResponse


class RecordingLimiter:
    """Stands in for ConcurrencyLimiter and records the order of events."""

    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def slot(self, provider: str):
        self.events.append(f"enter:{provider}")
        try:
            yield
        finally:
            self.events.append(f"exit:{provider}")


def _fake_backend(limiter, events):
    class Backend:
        def generate(self, **kwargs):
            events.append("call")
            return LLMResponse(text="{}", model="m", done=True, error=None)

    return Backend()


def test_the_slot_wraps_the_call_and_names_the_resolved_provider(monkeypatch):
    limiter = RecordingLimiter()
    monkeypatch.setattr(
        llm_router,
        "_resolve",
        lambda model, urls: (_fake_backend(limiter, limiter.events),
                             llm_router.split_model_id(model)[0], "bare"),
    )
    llm_router.generate(
        model="vllm:whatever",
        prompt="p",
        provider_urls={"vllm": "http://example.invalid"},
        limiter=limiter,
    )
    assert limiter.events == ["enter:vllm", "call", "exit:vllm"]


def test_the_slot_is_released_when_the_backend_raises(monkeypatch):
    limiter = RecordingLimiter()

    class Boom:
        def generate(self, **kwargs):
            raise RuntimeError("network died")

    monkeypatch.setattr(
        llm_router,
        "_resolve",
        lambda model, urls: (Boom(), llm_router.split_model_id(model)[0], "bare"),
    )
    try:
        llm_router.generate(
            model="vllm:whatever",
            prompt="p",
            provider_urls={"vllm": "http://example.invalid"},
            limiter=limiter,
        )
    except RuntimeError:
        pass
    assert limiter.events == ["enter:vllm", "exit:vllm"]


def test_without_a_limiter_nothing_changes(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        llm_router,
        "_resolve",
        lambda model, urls: (_fake_backend(None, events),
                             llm_router.split_model_id(model)[0], "bare"),
    )
    result = llm_router.generate(
        model="vllm:whatever", prompt="p", provider_urls={"vllm": "http://example.invalid"}
    )
    assert events == ["call"]
    assert result.error is None


def test_an_unconfigured_provider_never_takes_a_slot():
    limiter = RecordingLimiter()
    result = llm_router.generate(
        model="ds4:model", prompt="p", provider_urls={"ds4": ""}, limiter=limiter
    )
    assert result.error == "ds4: endpoint not configured"
    assert limiter.events == []
```

- [ ] **Step 2: Run it and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_llm_router_limiter.py -v`
Expected: FAIL — `generate()` got an unexpected keyword argument `limiter`.

- [ ] **Step 3: Wire the limiter into the router**

In `archiver/llm_router.py`:

Add `from contextlib import nullcontext` and, under `TYPE_CHECKING`, `from .concurrency import ConcurrencyLimiter`.

Add `limiter: Optional["ConcurrencyLimiter"] = None` as the last parameter of `generate`, and replace the final call:

```python
    # The slot wraps the request and nothing else: building a prompt does not
    # occupy the server, and holding a slot while doing it would waste it.
    ctx = limiter.slot(spec.name) if limiter is not None else nullcontext()
    with ctx:
        response = backend.generate(**kwargs)
    return _to_legacy(response, model=model)
```

Add the same parameter to `generate_with_image_file` and forward it in the inner `generate(...)` call.

Update the module docstring's last paragraph to:

```
The layer holds no state of its own: one backend per call, and the concurrency
limit arrives as an argument rather than living here, so a parallel scan needs
no coordination inside this module.
```

- [ ] **Step 4: Run the tests**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: PASS — including the pre-existing `tests/test_llm_router.py`, untouched, since `limiter` defaults to `None`.

- [ ] **Step 5: Commit**

```bash
git add archiver/llm_router.py tests/test_llm_router_limiter.py
git commit -m "feat: let the router hold a per-provider slot around each call"
```

---

### Task 4: Config and settings carry the limits

**Files:**
- Modify: `archiver/config.py`, `archiver/settings.py`, `archiver/setup_logic.py`, `archiver/__main__.py`
- Test: `tests/test_config_concurrency.py`

**Interfaces:**
- Consumes: `clamp_limit` (Task 2), `default_provider_concurrency` (Task 1)
- Produces: `AppConfig.provider_concurrency: dict[str, int]`, `Settings.provider_concurrency: dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_concurrency.py
import json

from archiver.config import AppConfig, load_config, save_config
from archiver.settings import Settings
from pathlib import Path


def test_a_config_written_before_0_13_gets_the_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "amenity-stuff" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"providers": {"vllm": "http://example.invalid"}}))
    cfg = load_config()
    assert cfg.provider_concurrency == {"vllm": 4, "ollama": 1, "ds4": 1}


def test_stored_values_win_and_are_clamped(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "amenity-stuff" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"provider_concurrency": {"vllm": 8, "ollama": 0, "ds4": 999}}))
    cfg = load_config()
    assert cfg.provider_concurrency == {"vllm": 8, "ollama": 1, "ds4": 16}


def test_unknown_provider_names_are_dropped():
    cfg = AppConfig(provider_concurrency={"vllm": 2, "ghost": 5})
    assert cfg.provider_concurrency == {"vllm": 2, "ollama": 1, "ds4": 1}


def test_settings_normalises_the_same_way():
    settings = Settings(
        source_root=Path("/tmp/src"),
        archive_root=Path("/tmp/arc"),
        provider_concurrency={"vllm": 6},
    )
    assert settings.provider_concurrency == {"vllm": 6, "ollama": 1, "ds4": 1}


def test_the_limits_survive_a_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_config(AppConfig(provider_concurrency={"vllm": 3}))
    assert load_config().provider_concurrency["vllm"] == 3
```

- [ ] **Step 2: Run it and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_config_concurrency.py -v`
Expected: FAIL — `AppConfig.__init__()` got an unexpected keyword argument.

- [ ] **Step 3: Add the field in both dataclasses**

In `archiver/config.py`: import `default_provider_concurrency` from `.providers` and `clamp_limit` from `.concurrency`; add the field after `providers`:

```python
    provider_concurrency: dict[str, int] = None  # type: ignore[assignment]
```

and append to `__post_init__`:

```python
        limits = default_provider_concurrency()
        if self.provider_concurrency:
            for name, value in self.provider_concurrency.items():
                if name in limits:
                    limits[name] = clamp_limit(value, default=limits[name])
        object.__setattr__(self, "provider_concurrency", limits)
```

In `load_config`, after the `providers` block:

```python
    concurrency = default_provider_concurrency()
    concurrency_raw = data.get("provider_concurrency")
    if isinstance(concurrency_raw, dict):
        for name, value in concurrency_raw.items():
            if name in PROVIDER_NAMES:
                concurrency[name] = clamp_limit(value, default=concurrency[name])
    kwargs["provider_concurrency"] = concurrency
```

Apply the identical field and `__post_init__` block to `archiver/settings.py`.

In `archiver/setup_logic.py`, `app_config_from_settings` must carry `provider_concurrency=dict(settings.provider_concurrency)`, and `settings_from_setup` must preserve it (it uses `replace`, so confirm no explicit provider list drops it).

In `archiver/__main__.py`, both `Settings(...)` constructions (around lines 70 and 91) gain `provider_concurrency=cfg.provider_concurrency`.

- [ ] **Step 4: Run the tests**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: PASS, whole suite.

- [ ] **Step 5: Commit**

```bash
git add archiver/config.py archiver/settings.py archiver/setup_logic.py archiver/__main__.py tests/test_config_concurrency.py
git commit -m "feat: persist the per-provider concurrency limits"
```

---

### Task 5: A field per provider in Settings

**Files:**
- Modify: `archiver/settings_screen.py`, `archiver/app.py`
- Test: `tests/test_settings_screen_concurrency.py`

**Interfaces:**
- Consumes: `clamp_limit`, `Settings.provider_concurrency`
- Produces: `SettingsResult.provider_concurrency: dict[str, int]`

- [ ] **Step 1: Write the failing test**

The screen itself is not driveable without a running app, so test the two pure pieces: the result dataclass carries the field, and parsing follows `clamp_limit`.

```python
# tests/test_settings_screen_concurrency.py
from pathlib import Path

from archiver.concurrency import clamp_limit
from archiver.providers import PROVIDERS
from archiver.settings_screen import SettingsResult


def test_result_carries_the_limits():
    result = SettingsResult(
        output_language="auto",
        taxonomies={},
        facts_model="auto",
        classify_model="auto",
        vision_model="auto",
        filename_separator="space",
        ocr_mode="balanced",
        undated_folder_name="undated",
        archive_root=Path("/tmp/arc"),
        providers={"vllm": "http://example.invalid"},
        provider_concurrency={"vllm": 4, "ollama": 1, "ds4": 1},
    )
    assert result.provider_concurrency["vllm"] == 4


def test_an_emptied_field_falls_back_to_the_registry_default():
    for spec in PROVIDERS:
        assert clamp_limit("", default=spec.max_concurrency) == spec.max_concurrency
```

- [ ] **Step 2: Run it and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_settings_screen_concurrency.py -v`
Expected: FAIL — `SettingsResult.__init__()` got an unexpected keyword argument.

- [ ] **Step 3: Add the field to the screen**

In `archiver/settings_screen.py`:

Add `provider_concurrency: dict[str, int]` to `SettingsResult`.

Import `from textual.containers import Horizontal` and `from .concurrency import clamp_limit`.

Add to `__init__` the parameter `provider_concurrency: dict[str, int]` and store:

```python
        self._concurrency = {
            spec.name: clamp_limit(
                (provider_concurrency or {}).get(spec.name), default=spec.max_concurrency
            )
            for spec in PROVIDERS
        }
```

In `compose`, replace the URL `Input` with a row holding both inputs:

```python
        for spec in PROVIDERS:
            placeholder = spec.default_url or "empty = disabled"
            yield Static(f"{spec.name} endpoint ({placeholder}):",
                         classes="provider_label", id=f"{spec.name}_label")
            with Horizontal(classes="provider_row"):
                yield Input(value=self._providers.get(spec.name, ""),
                            placeholder=placeholder,
                            classes="provider_url", id=f"{spec.name}_url")
                yield Input(value=str(self._concurrency[spec.name]),
                            placeholder="parallel",
                            classes="provider_par", id=f"{spec.name}_par")
```

Add to `CSS`:

```
    .provider_row { height: 3; }
    .provider_url { width: 1fr; }
    .provider_par { width: 14; }
```

Add the reader, mirroring `_current_provider_urls`:

```python
    def _current_provider_concurrency(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for spec in PROVIDERS:
            try:
                raw = self.query_one(f"#{spec.name}_par", Input).value.strip()
            except Exception:
                raw = ""
            out[spec.name] = clamp_limit(raw, default=self._concurrency[spec.name])
        return out
```

Pass `provider_concurrency=self._current_provider_concurrency()` in `action_save`, and `provider_concurrency=dict(self._concurrency)` in `action_cancel`.

In `archiver/app.py`:
- `action_settings` passes `provider_concurrency=self.settings.provider_concurrency` into `SettingsScreen(...)`.
- `_on_settings_done` adds `provider_concurrency=result.provider_concurrency` to the `replace(...)`. Leave `endpoints_changed` keyed on `providers` alone — changing a slot count does not warrant re-running discovery.

- [ ] **Step 4: Run the tests**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: PASS. `tests/test_settings_screen_result.py` may need the new keyword added to its constructions — update it if so.

- [ ] **Step 5: Verify the screen opens, under a PTY**

```bash
SCRATCH=$(mktemp -d)
script -qec "env XDG_CONFIG_HOME=$SCRATCH ~/.local/share/amenity-stuff/venv/bin/amenity-ai --source $SCRATCH --archive $SCRATCH" /dev/null
```

Press `F2`. Expected: each provider row shows the URL and a narrow `parallel` box; vLLM reads `4`. Save with `Ctrl+S`, reopen, and confirm the value persisted. A field collision or a layout error shows up here and nowhere else — the same class of bug as `DoctorScreen._render`.

- [ ] **Step 6: Commit**

```bash
git add archiver/settings_screen.py archiver/app.py tests/test_settings_screen_concurrency.py
git commit -m "tui: add a per-provider parallelism field to settings"
```

---

### Task 6: Thread the limiter through analysis

**Files:**
- Modify: `archiver/analyzer.py`, `archiver/task_builders.py`, `archiver/extractors/image.py`
- Test: `tests/test_analysis_limiter_propagation.py`

**Interfaces:**
- Consumes: `ConcurrencyLimiter`
- Produces: `AnalysisConfig.limiter`, `build_analysis_config(..., limiter=None)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_limiter_propagation.py
from pathlib import Path

from archiver import analyzer
from archiver.concurrency import ConcurrencyLimiter
from archiver.ollama_client import OllamaGenerateResult
from archiver.settings import Settings
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines
from archiver.task_builders import build_analysis_config


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


def test_the_facts_call_forwards_the_limiter_to_the_router(monkeypatch):
    seen: dict[str, object] = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return OllamaGenerateResult(response='{"summary_long": "x"}', model="m", done=True)

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
```

- [ ] **Step 2: Run it and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_analysis_limiter_propagation.py -v`
Expected: FAIL — unexpected keyword `limiter`.

- [ ] **Step 3: Thread it through**

In `archiver/analyzer.py`:
- add `from typing import TYPE_CHECKING` guarded import of `ConcurrencyLimiter`;
- add to `AnalysisConfig`: `limiter: Optional["ConcurrencyLimiter"] = None`;
- add a keyword-only `limiter: Optional["ConcurrencyLimiter"] = None` to `_extract_facts_from_text` and `_repair_json_dict_via_llm`, and pass `limiter=limiter` to every `generate(...)` inside them;
- in `extract_facts_item`, pass `limiter=config.limiter` to both `_extract_facts_from_text` call sites and to `extract_image_smart(...)`.

In `archiver/extractors/image.py`, add `limiter=None` to `caption_image` and `extract_image_smart`, and forward it to `generate_with_image_file(...)`.

In `archiver/task_builders.py`, add the keyword-only parameter to `build_analysis_config` and pass it into `AnalysisConfig`:

```python
def build_analysis_config(
    *,
    settings: "Settings",
    discovery: "DiscoveryResult | None",
    taxonomy: "Taxonomy",
    limiter: "ConcurrencyLimiter | None" = None,
) -> AnalysisConfig:
```

Also add `limiter` to `normalize_items` / `normalize_items_with_fallback` signatures now (keyword-only, default `None`, forwarded to `generate`) so phase 2 has nothing left to plumb — the classify path then honours the limit even while still running its chunks sequentially.

- [ ] **Step 4: Run the tests**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add archiver/analyzer.py archiver/task_builders.py archiver/extractors/image.py archiver/normalizer.py tests/test_analysis_limiter_propagation.py
git commit -m "feat: carry the concurrency limiter down the analysis path"
```

---

### Task 7: Throttle cache writes

**Files:**
- Modify: `archiver/cache.py`
- Test: `tests/test_cache_throttle.py`

**Interfaces:**
- Produces: `SaveThrottle(min_interval_s=5.0, min_dirty=25, clock=time.monotonic)` with `.record() -> bool` and `.flush() -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_throttle.py
from archiver.cache import SaveThrottle


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_it_fires_on_the_count_trigger():
    clock = FakeClock()
    throttle = SaveThrottle(min_interval_s=1000.0, min_dirty=3, clock=clock)
    assert throttle.record() is False
    assert throttle.record() is False
    assert throttle.record() is True
    assert throttle.record() is False


def test_it_fires_on_the_time_trigger():
    clock = FakeClock()
    throttle = SaveThrottle(min_interval_s=5.0, min_dirty=1000, clock=clock)
    assert throttle.record() is False
    clock.now = 5.0
    assert throttle.record() is True


def test_flush_writes_only_what_is_pending():
    clock = FakeClock()
    throttle = SaveThrottle(min_interval_s=1000.0, min_dirty=1000, clock=clock)
    assert throttle.flush() is False, "nothing changed, nothing to write"
    throttle.record()
    assert throttle.flush() is True
    assert throttle.flush() is False
```

- [ ] **Step 2: Run it and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_cache_throttle.py -v`
Expected: FAIL — cannot import `SaveThrottle`.

- [ ] **Step 3: Add it to `archiver/cache.py`**

```python
class SaveThrottle:
    """Decide when a cache write is worth its cost.

    One write re-serialises the whole cache. At one file every twenty seconds
    that is invisible; with several in flight it starts stealing frames from
    the event loop, which is where the write happens.
    """

    def __init__(
        self,
        *,
        min_interval_s: float = 5.0,
        min_dirty: int = 25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._min_dirty = min_dirty
        self._clock = clock
        self._dirty = 0
        self._last = clock()

    def record(self) -> bool:
        """Register one changed entry. True when it is time to write."""
        self._dirty += 1
        due = (
            self._dirty >= self._min_dirty
            or (self._clock() - self._last) >= self._min_interval_s
        )
        if due:
            self._reset()
        return due

    def flush(self) -> bool:
        """True when unwritten changes remain. Call at the end of a run."""
        if self._dirty == 0:
            return False
        self._reset()
        return True

    def _reset(self) -> None:
        self._dirty = 0
        self._last = self._clock()
```

Add `import time` and `from typing import Callable` if absent.

- [ ] **Step 4: Run the tests**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_cache_throttle.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add archiver/cache.py tests/test_cache_throttle.py
git commit -m "feat: throttle cache writes so a fast run cannot stall the ui"
```

---

### Task 8: Progress, rate and ETA as pure functions

**Files:**
- Modify: `archiver/ui_status.py`
- Test: `tests/test_ui_progress.py`

**Interfaces:**
- Produces: `RunProgress`, `compute_rate`, `format_eta`, `progress_line`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_progress.py
from archiver.ui_status import RunProgress, compute_rate, format_eta, progress_line


def test_queued_is_what_is_neither_done_nor_running():
    p = RunProgress(total=100, completed=10, in_flight=4)
    assert p.queued == 86


def test_queued_never_goes_negative():
    assert RunProgress(total=2, completed=2, in_flight=4).queued == 0


def test_rate_needs_two_completions():
    assert compute_rate([], now=100.0) is None
    assert compute_rate([95.0], now=100.0) is None


def test_rate_counts_intervals_not_events():
    # two completions ten seconds apart is one per ten seconds
    assert compute_rate([90.0, 100.0], now=100.0) == 0.1


def test_rate_ignores_completions_outside_the_window():
    # the old one is dropped, leaving a single sample
    assert compute_rate([0.0, 95.0], now=100.0, window_s=60.0) is None


def test_eta_formats_by_magnitude():
    assert format_eta(None) == ""
    assert format_eta(0) == ""
    assert format_eta(45) == "~45s left"
    assert format_eta(600) == "~10m left"
    assert format_eta(3840) == "~1h04m left"


def test_progress_line_without_a_rate_omits_the_estimate():
    line = progress_line(
        RunProgress(total=810, completed=12, in_flight=4), rate=None, total_files=810
    )
    assert line == (
        "files: 810 • queued: 794 • in flight: 4 • done: 12 • skipped: 0 • error: 0"
    )


def test_progress_line_with_a_rate_shows_throughput_and_eta():
    line = progress_line(
        RunProgress(total=810, completed=12, in_flight=4), rate=0.2, total_files=810
    )
    assert "0.20 file/s" in line
    # 798 left at 0.2/s is 3990s
    assert "~1h06m left" in line
```

- [ ] **Step 2: Run it and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_ui_progress.py -v`
Expected: FAIL — cannot import `RunProgress`.

- [ ] **Step 3: Implement in `archiver/ui_status.py`**

```python
@dataclass(frozen=True)
class RunProgress:
    """Progress of one run, not of the table.

    A table can hold files classified or moved in an earlier session; counting
    those as this run's progress is what produces numbers that do not add up.
    """

    total: int
    completed: int
    in_flight: int
    skipped: int = 0
    error: int = 0

    @property
    def queued(self) -> int:
        return max(0, self.total - self.completed - self.in_flight)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)


def compute_rate(
    timestamps: Sequence[float], *, now: float, window_s: float = 60.0
) -> Optional[float]:
    """Completions per second over the trailing window.

    A sliding window rather than the run average, so one slow file shows up
    instead of being diluted by everything that came before it.
    """
    recent = sorted(t for t in timestamps if now - t <= window_s)
    if len(recent) < 2:
        return None
    span = recent[-1] - recent[0]
    if span <= 0:
        return None
    return (len(recent) - 1) / span


def format_eta(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return ""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"~{hours}h{minutes:02d}m left"
    if minutes:
        return f"~{minutes}m left"
    return f"~{secs}s left"


def progress_line(
    progress: RunProgress, *, rate: Optional[float], total_files: int
) -> str:
    bits = [
        f"files: {total_files}",
        f"queued: {progress.queued}",
        f"in flight: {progress.in_flight}",
        f"done: {progress.completed}",
        f"skipped: {progress.skipped}",
        f"error: {progress.error}",
    ]
    if rate:
        bits.append(f"{rate:.2f} file/s")
        eta = format_eta(progress.remaining / rate)
        if eta:
            bits.append(eta)
    return " • ".join(bits)
```

Add `from dataclasses import dataclass` and `from typing import Optional, Sequence` to the imports.

- [ ] **Step 4: Run the tests**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_ui_progress.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add archiver/ui_status.py tests/test_ui_progress.py
git commit -m "feat: compute run progress, throughput and eta"
```

---

### Task 9: A banner that judges roles, not one provider

**Files:**
- Modify: `archiver/ui_runtime.py`, `archiver/app.py`
- Test: `tests/test_ui_runtime_problem.py`

**Interfaces:**
- Consumes: `run_doctor` from `archiver/doctor.py`
- Produces: `runtime_problem(discovery, settings) -> tuple[str | None, str]`; `provider_problem` is removed

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_runtime_problem.py
from pathlib import Path

from archiver.capabilities import CAP_VISION, SOURCE_DECLARED
from archiver.discovery import DiscoveryResult, ModelInfo, ProviderStatus
from archiver.settings import Settings
from archiver.ui_runtime import banner_for_state, runtime_problem


def _settings() -> Settings:
    return Settings(source_root=Path("/tmp/s"), archive_root=Path("/tmp/a"))


def _model(mid: str, *, vision: bool) -> ModelInfo:
    return ModelInfo(
        id=mid,
        provider=mid.split(":")[0],
        capabilities=frozenset({CAP_VISION}) if vision else frozenset(),
        capability_source=SOURCE_DECLARED,
        parameter_size_b=27.0,
    )


def _discovery(*statuses: ProviderStatus) -> DiscoveryResult:
    return DiscoveryResult(providers=tuple(statuses))


def test_no_discovery_yet_is_informational():
    assert runtime_problem(None, _settings()) == ("Detecting providers…", "info")


def test_a_covered_setup_is_clean():
    ok = ProviderStatus(
        name="vllm", url="http://example.invalid", configured=True, available=True,
        models=(_model("vllm:m", vision=True),),
    )
    assert runtime_problem(_discovery(ok), _settings()) == (None, "ok")


def test_a_provider_that_is_down_only_warns_when_the_roles_are_covered():
    ok = ProviderStatus(
        name="vllm", url="http://example.invalid", configured=True, available=True,
        models=(_model("vllm:m", vision=True),),
    )
    down = ProviderStatus(
        name="ollama", url="http://example.invalid", configured=True, available=False,
        detail="URLError", models=(),
    )
    message, severity = runtime_problem(_discovery(ok, down), _settings())
    assert severity == "warn"
    assert "ollama" in message


def test_no_semantic_model_is_an_error():
    empty = ProviderStatus(
        name="vllm", url="http://example.invalid", configured=True, available=True, models=(),
    )
    message, severity = runtime_problem(_discovery(empty), _settings())
    assert severity == "error"
    assert message == "No semantic model available"


def test_a_warning_does_not_replace_the_running_banner():
    text, _ = banner_for_state(
        state="scanning…", scanning=4, classifying=0, moving=0,
        problem="ollama unreachable", severity="warn",
    )
    assert "RUNNING" in text
    assert "ollama unreachable" in text


def test_the_running_banner_says_how_many_are_in_flight():
    text, _ = banner_for_state(
        state="scanning…", scanning=4, classifying=0, moving=0, problem=None, severity="ok",
    )
    assert "4 in flight" in text


def test_stopping_says_how_many_it_is_waiting_for():
    text, _ = banner_for_state(
        state="stopping…", scanning=3, classifying=0, moving=0, problem=None, severity="ok",
    )
    assert text == "STOPPING — waiting for 3 requests in flight"
```

> Check the real constructor signatures of `ProviderStatus`, `ModelInfo` and `DiscoveryResult` in `archiver/discovery.py` before running, and adjust the helpers to match — these are the shapes as of 0.12.3.

- [ ] **Step 2: Run it and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_ui_runtime_problem.py -v`
Expected: FAIL — cannot import `runtime_problem`.

- [ ] **Step 3: Replace `provider_problem`**

In `archiver/ui_runtime.py`, delete `provider_problem` and add:

```python
def runtime_problem(
    discovery: "DiscoveryResult | None", settings: "Settings"
) -> tuple[str | None, str]:
    """Severity of the current setup, judged by roles rather than by one provider.

    Reuses the doctor — same logic, two surfaces. `probe=None` keeps it pure:
    drawing a banner must never open a socket.
    """
    if not discovery:
        return ("Detecting providers…", "info")

    from .doctor import STATUS_FAIL, STATUS_WARN, run_doctor

    report = run_doctor(discovery=discovery, settings=settings, probe=None)
    by_key = {c.key: c for c in report.checks}
    text = by_key.get("role.text")
    vision = by_key.get("role.vision")

    if text is not None and text.status == STATUS_FAIL:
        return ("No semantic model available", "error")
    if vision is not None and vision.status == STATUS_FAIL:
        return ("No vision model — images will be skipped", "warn")
    for check in report.checks:
        if check.key.startswith("provider.") and check.status == STATUS_FAIL:
            return (f"{check.key.split('.', 1)[1]} unreachable", "warn")
    for check in (text, vision):
        if check is not None and check.status == STATUS_WARN:
            return (check.detail or f"{check.label}: warning", "warn")
    return (None, "ok")
```

In `banner_for_state`, after the `severity == "error"` branch, add:

```python
    if severity == "warn" and state == "idle":
        return (f"WARNING: {problem}", "bold black on yellow")
```

Change the stopping branch to:

```python
    if state.startswith("stopping"):
        in_flight = scanning + classifying + moving
        if in_flight:
            return (f"STOPPING — waiting for {in_flight} requests in flight",
                    "bold white on red")
        return ("STOPPING…", "bold white on red")
```

and the scanning branch's message to `f"RUNNING: scanning — {scanning} in flight"`.

In `archiver/app.py`, change the import to `runtime_problem` and the call in `_render_notes` to `runtime_problem(self._discovery, self.settings)`.

- [ ] **Step 4: Run the tests**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: PASS. Any existing test referencing `provider_problem` must be deleted along with it — its behaviour no longer exists.

- [ ] **Step 5: Commit**

```bash
git add archiver/ui_runtime.py archiver/app.py tests/test_ui_runtime_problem.py
git commit -m "fix: judge the banner by roles instead of by ollama alone"
```

---

### Task 10: The parallel facts worker

**Files:**
- Modify: `archiver/app.py` (`__init__`, `_run_extract_pending`, `_render_notes`)

This is the integration task: it has no unit test of its own — every piece it composes is already tested, and the wiring is verified by hand under a PTY.

- [ ] **Step 1: Add run-progress state**

In `ArchiverApp.__init__`:

```python
        self._run_total = 0
        self._run_completed = 0
        self._run_in_flight = 0
        self._run_skipped = 0
        self._run_error = 0
        self._run_completions: list[float] = []
        self._cache_throttle = SaveThrottle()
```

Import `SaveThrottle` from `.cache`, `RunProgress`, `compute_rate`, `progress_line` from `.ui_status`, `ConcurrencyLimiter`, `pool_size_for` from `.concurrency`, and `from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed`.

- [ ] **Step 2: Rewrite the body of `_run_extract_pending`**

Keep the guard clauses at the top of the method as they are; replace the inner callbacks and the worker function with these — `mark_scanning` and `apply_result` gain the progress bookkeeping, and `begin_run` is new:

```python
        def begin_run(total: int) -> None:
            self._run_total = total
            self._run_completed = 0
            self._run_in_flight = 0
            self._run_skipped = 0
            self._run_error = 0
            self._run_completions.clear()
            self._cache_throttle = SaveThrottle()
            self._render_notes()

        def mark_scanning(path_str: str) -> None:
            idx = self._scan_index_by_path.get(path_str)
            if idx is None:
                return
            it = self._scan_items[idx]
            if it.status != "pending":
                return
            self._scan_items[idx] = mark_item_scanning(it)
            self._run_in_flight += 1
            files.update_cell(path_str, "status", status_cell("scanning"))
            self._render_notes()
            if files.cursor_row == idx:
                self._update_details(idx)

        def apply_result(path_str: str, new_item: ScanItem) -> None:
            idx = self._scan_index_by_path.get(path_str)
            if idx is None:
                return
            self._scan_items[idx] = new_item
            self._run_in_flight = max(0, self._run_in_flight - 1)
            self._run_completed += 1
            if new_item.status == "skipped":
                self._run_skipped += 1
            elif new_item.status == "error":
                self._run_error += 1
            self._run_completions.append(time.monotonic())
            files.update_cell(path_str, "status", status_cell(new_item.status))
            files.update_cell(path_str, "category", new_item.category or "")
            files.update_cell(path_str, "year", new_item.reference_year or "")
            self._render_notes()
            if files.cursor_row == idx:
                self._update_details(idx)
            if self._cache:
                self._cache.upsert(new_item)
                # A write re-serialises the whole cache; at four files in
                # flight, doing it per file would stall the event loop.
                if self._cache_throttle.record():
                    self._cache.save()

        def finish(cancelled: bool) -> None:
            if cancelled:
                for idx, it in enumerate(list(self._scan_items)):
                    if it.status != "scanning":
                        continue
                    updated = replace(it, status="pending", reason="Scan stopped")
                    self._scan_items[idx] = updated
                    files.update_cell(str(updated.path), "status", status_cell("pending"))
            if self._cache and self._cache_throttle.flush():
                self._cache.save()
            self._run_in_flight = 0
            self._analysis_task.running = False
            self._render_notes()

        def do_extract_background() -> None:
            worker = get_current_worker()
            taxonomy, _ = parse_taxonomy_lines(self.settings.get_taxonomy_lines())
            limiter = ConcurrencyLimiter.from_limits(self.settings.provider_concurrency)
            cfg = build_analysis_config(
                settings=self.settings, discovery=self._discovery,
                taxonomy=taxonomy, limiter=limiter,
            )
            targets = [it for it in self._scan_items if it.status == "pending"]
            self.call_from_thread(begin_run, len(targets))

            def run_one(it: ScanItem):
                # Checked here as well as in the loop below: a thread that has
                # just finished an item would otherwise pick up the next one
                # before the executor is told to stop.
                if worker.is_cancelled:
                    return None
                path_str = str(it.path)
                self.call_from_thread(mark_scanning, path_str)
                t0 = time.perf_counter()
                try:
                    res = extract_facts_item(it, config=cfg)
                except Exception as exc:  # noqa: BLE001
                    # One bad file must not take the run down with it: an
                    # exception escaping here would leave finish() uncalled and
                    # the UI stuck on "running" forever.
                    return path_str, replace(
                        it, status="error", reason=f"Scan crashed: {type(exc).__name__}"
                    )
                elapsed = time.perf_counter() - t0
                return path_str, replace(
                    it,
                    status=res.status,
                    reason=res.reason,
                    confidence=res.confidence,
                    analysis_time_s=elapsed,
                    model_used=res.model_used,
                    summary_long=res.summary_long,
                    facts_json=res.facts_json,
                    llm_raw_output=res.llm_raw_output,
                    extract_method=res.extract_method,
                    extract_time_s=res.extract_time_s,
                    llm_time_s=res.llm_time_s,
                    ocr_time_s=res.ocr_time_s,
                    ocr_mode=res.ocr_mode,
                    facts_time_s=elapsed,
                    facts_llm_time_s=res.llm_time_s,
                    facts_model_used=res.model_used,
                    category=None,
                    reference_year=None,
                    proposed_name=None,
                    summary=None,
                )

            executor = ThreadPoolExecutor(
                max_workers=pool_size_for(self.settings.providers, limiter),
                thread_name_prefix="facts",
            )
            stopped = False
            try:
                futures = [executor.submit(run_one, it) for it in targets]
                for future in as_completed(futures):
                    if worker.is_cancelled and not stopped:
                        stopped = True
                        # Drop what is queued; what is already running finishes
                        # and is still applied, so no completed work is thrown away.
                        executor.shutdown(wait=False, cancel_futures=True)
                    try:
                        outcome = future.result()
                    except CancelledError:
                        continue
                    except Exception:  # noqa: BLE001
                        continue
                    if outcome is None:
                        continue
                    path_str, updated = outcome
                    self.call_from_thread(apply_result, path_str, updated)
            finally:
                executor.shutdown(wait=True)
            self.call_from_thread(finish, worker.is_cancelled)
```

Note `_run_extract_row` keeps its own single-item path unchanged — one file needs no pool. Give it `limiter=ConcurrencyLimiter.from_limits(self.settings.provider_concurrency)` anyway, so a manual `s` during no other work still respects a provider that only wants one caller.

- [ ] **Step 3: Show progress while a run is live**

In `_render_notes`, replace the `notes_line(...)` call with:

```python
        if self._analysis_task.running and self._run_total:
            progress = RunProgress(
                total=self._run_total,
                completed=self._run_completed,
                in_flight=self._run_in_flight,
                skipped=self._run_skipped,
                error=self._run_error,
            )
            text = progress_line(
                progress,
                rate=compute_rate(self._run_completions, now=time.monotonic()),
                total_files=counts.total,
            )
        else:
            text = notes_line(
                scan_items_total=counts.total,
                pending=counts.pending,
                scanning=counts.scanning,
                scanned=counts.scanned,
                classifying=counts.classifying,
                classified=counts.classified,
                moved=counts.moved,
                skipped=counts.skipped,
                error=counts.error,
            )
        self.query_one("#notes", Static).update(text)
```

- [ ] **Step 4: Run the suite**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: PASS.

- [ ] **Step 5: Verify against the real server, under a PTY**

```bash
SCRATCH=$(mktemp -d) && mkdir -p $SCRATCH/src $SCRATCH/arc
cp <about 40 mixed pdfs and images> $SCRATCH/src/
cp ~/.config/amenity-stuff/config.json $SCRATCH/amenity-stuff/config.json   # endpoints only
script -qec "env XDG_CONFIG_HOME=$SCRATCH ~/.local/share/amenity-stuff/venv/bin/amenity-ai --source $SCRATCH/src --archive $SCRATCH/arc" /dev/null
```

Press `S` and check, in order:

1. four rows sit in `scanning` at once, not one;
2. the notes line reads `queued / in flight / done` and the rate and ETA appear once two files are done;
3. the banner reads `RUNNING: scanning — 4 in flight` and is **not** red (Ollama is down but vLLM covers both roles);
4. `x` switches the banner to `STOPPING — waiting for N requests in flight`, the count falls to zero, and rows still running go back to `pending`;
5. total wall clock for the set is roughly a third of the same run with `vllm` set to `parallel: 1` in `F2` — that comparison is the acceptance criterion, and it is worth running both ways once.

- [ ] **Step 6: Commit**

```bash
git add archiver/app.py
git commit -m "feat: run the facts phase in parallel, bounded per provider"
```

---

### Task 11: Delete the dead Ollama entry points

**Files:**
- Modify: `archiver/ollama_client.py`

- [ ] **Step 1: Confirm they have no callers**

```bash
grep -rn "ollama_client" archiver/ tests/ | grep -v "^archiver/ollama_client.py"
```

Expected: only `OllamaBackend` and `OllamaGenerateResult` are imported. If anything imports `ollama_client.generate`, stop and reconsider — the premise of this task is false.

- [ ] **Step 2: Delete**

Remove `_default_backend`, `_get_backend`, the module-level `generate` and the module-level `generate_with_image_file`, along with the `# Backward-compatible module-level functions` comment. Keep `OllamaBackend`, `OllamaGenerateResult`, `_post_json` and `DEFAULT_BASE_URL`. Drop the `base64` import if nothing else uses it.

- [ ] **Step 3: Run the suite and import the app**

```bash
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v
~/.local/share/amenity-stuff/venv/bin/python -c "import archiver.app, archiver.__main__"
```

Expected: PASS, and both imports succeed.

- [ ] **Step 4: Commit**

```bash
git add archiver/ollama_client.py
git commit -m "chore: drop the unreachable module-level ollama entry points"
```

---

### Task 12: Version, changelog, docs

**Files:**
- Modify: `VERSION`, `pyproject.toml`, `CHANGELOG.md`, `README.md`, `CLAUDE.md`

- [ ] **Step 1: Bump to 0.13.0**

Set `0.13.0` in `VERSION` and in `pyproject.toml`'s `version =`. They must match.

- [ ] **Step 2: Changelog**

Add a `## 0.13.0` section following the existing style: the facts phase now runs several files at once bounded per provider; a per-provider parallelism field in Settings; the status line reports queued / in flight / done with throughput and ETA; the banner judges the setup by role instead of by Ollama alone; cache writes are throttled.

- [ ] **Step 3: README**

In the providers table add the concurrency default per provider, and note that it is editable in Settings. Mention that the number for vLLM was measured, not guessed.

- [ ] **Step 4: CLAUDE.md**

Under "Task Orchestration", replace "One long-running task (scan/classify/move) at a time" with a note that it is still one task at a time, but the facts phase runs its files through a pool bounded by `ConcurrencyLimiter`, one semaphore per provider, acquired in `llm_router.generate`. Add `archiver/concurrency.py` to the Module Organization list.

- [ ] **Step 5: Reinstall and check the version**

```bash
~/.local/share/amenity-stuff/venv/bin/pip install -e . 2>&1 | tail -5
~/.local/share/amenity-stuff/venv/bin/pip show amenity-ai | head -4
```

Expected: `Version: 0.13.0`. Read the install output in full — a failure here is silent and leaves the command on the old metadata.

- [ ] **Step 6: Commit and open the PR**

```bash
git add VERSION pyproject.toml CHANGELOG.md README.md CLAUDE.md
git commit -m "docs: document parallel scanning; release 0.13.0"
git push -u origin feat/parallel-scan
gh pr create --title "Parallel scanning, bounded per provider" --body-file <path>
```

The PR body is written in English, states the measured benchmark table, and names what was verified by hand rather than by tests. No Claude attribution anywhere.

---

# Phase 2 — parallel classify chunks (PR 2)

### Task 13: Chunks run through the same limiter

**Files:**
- Modify: `archiver/normalizer.py`
- Test: `tests/test_normalizer_parallel.py`

Branch from `main` after PR 1 lands: `git checkout main && git pull && git checkout -b feat/parallel-classify`.

**Interfaces:**
- Consumes: `ConcurrencyLimiter`, and the `limiter` parameter added to `normalize_items` in Task 6
- Produces: unchanged public signature; `normalize_items` runs its chunks concurrently

The delicate part is error semantics. Today: a failing chunk of more than one item falls back to single items and `continue`s; a failing single-item chunk `return`s immediately and abandons every chunk after it. "Return immediately" has no meaning once chunks run at once. New rule: **each chunk is independent**, results merge in input order, and the reported error is the first error — in input order — among chunks that produced nothing. A cancelled run still reports `"Cancelled"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalizer_parallel.py
import threading

from archiver import normalizer
from archiver.concurrency import ConcurrencyLimiter
from archiver.ollama_client import OllamaGenerateResult


def _items(n):
    from archiver.scanner import ScanItem
    from pathlib import Path
    return [
        ScanItem(path=Path(f"/tmp/f{i}.pdf"), kind="pdf", size_bytes=1,
                 mtime_iso="2026-01-01T00:00:00", status="scanned",
                 summary_long=f"doc {i}", facts_json="{}")
        for i in range(n)
    ]


def test_chunks_run_concurrently(monkeypatch):
    from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

    taxonomy, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)
    concurrent_peak = 0
    live = 0
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=5)

    def fake_generate(**kwargs):
        nonlocal concurrent_peak, live
        with lock:
            live += 1
            concurrent_peak = max(concurrent_peak, live)
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
        with lock:
            live -= 1
        return OllamaGenerateResult(response="[]", model="m", done=True)

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    normalizer.normalize_items(
        items=_items(4), model="vllm:m",
        provider_urls={"vllm": "http://example.invalid"},
        taxonomy=taxonomy, output_language="en", filename_separator="space",
        chunk_size=2,
        limiter=ConcurrencyLimiter.from_limits({"vllm": 2}),
    )
    assert concurrent_peak == 2


def test_one_failing_chunk_does_not_abandon_the_others(monkeypatch):
    from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

    taxonomy, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)
    calls = {"n": 0}

    def fake_generate(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return OllamaGenerateResult(response="", model="m", done=False, error="boom")
        return OllamaGenerateResult(
            response='[{"path": "/tmp/f2.pdf", "category": "personal", '
                     '"reference_year": "2024", "proposed_name": "n", "summary": "s"}]',
            model="m", done=True,
        )

    monkeypatch.setattr(normalizer, "generate", fake_generate)
    result = normalizer.normalize_items(
        items=_items(4), model="vllm:m",
        provider_urls={"vllm": "http://example.invalid"},
        taxonomy=taxonomy, output_language="en", filename_separator="space",
        chunk_size=2,
    )
    assert result.by_path, "the healthy chunk must still produce output"
```

> These tests depend on `ScanItem`'s real field names and on `_chunk`'s behaviour; check both before running and adjust the fixture rather than the production code.

- [ ] **Step 2: Run and watch it fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_normalizer_parallel.py -v`
Expected: FAIL — `concurrent_peak == 1`, because chunks are sequential.

- [ ] **Step 3: Restructure the loop**

Extract the body of the current `for batch in _chunk(items, chunk_size):` loop into a function `_normalize_one_chunk(batch, index, ...) -> tuple[int, dict, Optional[str]]` returning `(index, by_path_partial, error)`, keeping the existing per-chunk fallback to single items intact inside it. Then:

```python
    chunks = _chunk(items, chunk_size)
    limit = limiter.limit(split_model_id(model)[0].name) if limiter else 1
    with ThreadPoolExecutor(max_workers=max(1, min(len(chunks), limit))) as pool:
        futures = [pool.submit(_normalize_one_chunk, batch, i, ...) 
                   for i, batch in enumerate(chunks)]
        outcomes = sorted((f.result() for f in futures), key=lambda o: o[0])
    by_path: dict = {}
    first_error: Optional[str] = None
    for _, partial, error in outcomes:
        by_path.update(partial)
        if error and first_error is None and not partial:
            first_error = error
    return NormalizationResult(by_path=by_path, model_used=model, error=first_error)
```

Cancellation stays as it is — checked before each chunk starts, and `"Cancelled"` takes precedence over any other error.

The pool is bounded by the limiter's own number so the classify phase cannot exceed what the facts phase is allowed; the limiter is still acquired inside the router, so this is belt and braces rather than the enforcement point.

- [ ] **Step 4: Run the suite**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: PASS, including the pre-existing `tests/test_normalizer_routing.py`.

- [ ] **Step 5: Verify against the real server, under a PTY**

Same scratch setup as Task 10. Scan the set, then press `C` and confirm classifications are unchanged in content and materially faster; confirm `x` during classify still stops cleanly.

- [ ] **Step 6: Version, changelog, commit, PR**

Bump patch to `0.13.1` in `VERSION` and `pyproject.toml`, add the changelog entry, commit as `feat: run classification chunks in parallel`, push and open the PR in English.

---

## Self-review notes

- Every task ends green: no task leaves the suite red on purpose. If one does, combine it with the next rather than committing a broken tree, and say so in the commit message.
- Task 9 deletes `provider_problem`. Any test that referenced it goes with it — its behaviour is gone by design, not by accident.
- Task 6 adds `limiter` to the normalizer signatures even though phase 2 is what uses them concurrently. That is deliberate: it makes phase 2 a change to one function rather than a change spread across three modules.
- The benchmark script used to size the limits lives in the session scratchpad, not in the repo — it reads the user's real endpoint from their config and must not be committed.
