# ds4 OpenAI-Compatible Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second local LLM provider — an OpenAI-compatible chat-completions server ("ds4") — routed by the `ds4:` model-id prefix, coexisting with Ollama.

**Architecture:** A new `Ds4Backend` implements the existing `LLMBackend` protocol over `/v1/chat/completions`; a new `llm_router` module exposes the same module-level `generate()` API that `analyzer.py` already uses and dispatches on the `ds4:` prefix. Discovery, model selection, config, and the settings screen learn about the new provider; everything else is untouched.

**Tech Stack:** Python 3.10+, stdlib `urllib` (no new runtime dependencies), Textual TUI, pytest (dev-only, new `tests/` directory).

**Spec:** `docs/superpowers/specs/2026-08-01-ds4-openai-backend-design.md`

## Global Constraints

- Routing prefix is exactly `ds4:` (constant `DS4_PREFIX` in `archiver/llm_router.py`; everyone else imports it).
- Default endpoint is `""` (empty = feature disabled) in `AppConfig`, `Settings`, and `AnalysisConfig`. **Never** hardcode a real hostname; docs/examples use `http://localhost:8000`.
- Before EVERY commit run `sh scripts/check_no_private_host.sh` (created in Task 1) — it must print `OK`. The user's personal ds4 endpoint/hostname lives only in the local config (`~/.config/amenity-stuff/config.json`) and must never enter the repo: not in code, docs, tests, plan files, or git history. Do not write the hostname anywhere in the repository, including this plan and commit messages.
- ds4 is text-only: `ds4:*` models never appear in vision candidate lists; `images_b64` on the ds4 path returns an error `LLMResponse`.
- ds4 request mapping (from the spec, verified against the real server): `think=False` → `reasoning_effort: "low"`; fixed `max_tokens: 1500` (ignore `num_predict`); `temperature` passed through from `options`; `response_format` and `keep_alive` NOT sent; read ONLY `choices[0].message.content` (never `reasoning_content`).
- No new runtime dependencies. `pytest` is dev-only and is NOT added to `pyproject.toml`.
- Version: do NOT bump per task. One minor bump `0.9.13 → 0.10.0` (VERSION + pyproject.toml by hand) + CHANGELOG entry in the final task, as approved in the spec.
- Commit messages: `type: description`, no Co-Authored-By lines, no session URLs.
- Tests run with: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v` (install once: `~/.local/share/amenity-stuff/venv/bin/pip install pytest`). If that venv is missing, create it per README manual install first.
- Errors never raise across the TUI: all failure paths return `LLMResponse(error=...)` / `OllamaGenerateResult(error=...)`.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `archiver/openai_client.py` | Create | `Ds4Backend`: HTTP + payload/response mapping for chat completions |
| `archiver/llm_router.py` | Create | `DS4_PREFIX`, module-level `generate()`/`generate_with_image_file()` dispatching by prefix |
| `archiver/analyzer.py` | Modify | Import router instead of ollama_client; thread `ds4_base_url` |
| `archiver/discovery.py` | Modify | `_discover_ds4()`, `discover_providers(ds4_base_url=...)` |
| `archiver/model_selection.py` | Modify | ds4 models in text candidates (first), excluded from vision |
| `archiver/config.py` | Modify | `AppConfig.ds4_base_url` + defensive parsing |
| `archiver/settings.py` | Modify | `Settings.ds4_base_url` |
| `archiver/setup_logic.py` | Modify | Thread field through both conversion helpers |
| `archiver/__main__.py` | Modify | Wire `cfg.ds4_base_url` into `Settings` |
| `archiver/task_builders.py` | Modify | ds4-first `prefer_fast`; thread `ds4_base_url` into `AnalysisConfig` |
| `archiver/app.py` | Modify | Discovery arg, classify ordering, settings screen wiring, re-discovery on endpoint change |
| `archiver/settings_screen.py` | Modify | `Input` field for endpoint; `SettingsResult.ds4_base_url`; vision filter guard |
| `archiver/ui_status.py` | Modify | `provider_summary` shows both providers |
| `tests/…` | Create | pytest unit tests per module (no network, monkeypatched HTTP) |
| `README.md`, `CLAUDE.md`, `AGENTS.md`, `CHANGELOG.md`, `VERSION`, `pyproject.toml` | Modify | Docs + single minor version bump (final task) |

---

### Task 1: `Ds4Backend` (`archiver/openai_client.py`)

**Files:**
- Create: `archiver/openai_client.py`
- Create: `scripts/check_no_private_host.sh` (privacy guard used by every task's commit step)
- Test: `tests/test_openai_client.py`

**Interfaces:**
- Consumes: `BaseLLMBackend`, `LLMResponse` from `archiver/llm_backend.py` (existing).
- Produces: `class Ds4Backend(BaseLLMBackend)` with the protocol `generate()` signature (`prompt`, `model`, `timeout_s`, `images_b64`, `response_format`, `think`, `keep_alive`, `options` — all keyword-only) returning `LLMResponse`. Module-level `_post_json(url, payload, *, timeout_s)` (monkeypatch point for tests). Constant `_MAX_TOKENS = 1500`.

- [ ] **Step 0: Create the privacy guard script**

Create `scripts/check_no_private_host.sh` (generic: reads the user's private hostname from the LOCAL config, which is outside the repo — the hostname itself must never appear in this script or anywhere in the repo):

```sh
#!/bin/sh
# Fails if the user's private ds4 hostname (read from the local config,
# which lives outside the repo) appears anywhere in tracked files.
HOST=$(python3 -c '
import json, os
from urllib.parse import urlparse
path = os.path.expanduser("~/.config/amenity-stuff/config.json")
try:
    with open(path) as fh:
        cfg = json.load(fh)
    print(urlparse(cfg.get("ds4_base_url", "")).hostname or "")
except Exception:
    print("")
')
if [ -z "$HOST" ] || [ "$HOST" = "localhost" ] || [ "$HOST" = "127.0.0.1" ]; then
    echo "OK (no private host configured)"
    exit 0
fi
if git grep -qiF "$HOST" -- . 2>/dev/null; then
    echo "LEAK: private hostname found in tracked files:"
    git grep -inF "$HOST" -- .
    exit 1
fi
echo "OK"
```

Then: `chmod +x scripts/check_no_private_host.sh`

- [ ] **Step 1: Install pytest (once) and write the failing tests**

```bash
~/.local/share/amenity-stuff/venv/bin/pip install pytest
mkdir -p tests
```

Create `tests/test_openai_client.py`:

```python
from __future__ import annotations

import pytest

from archiver import openai_client
from archiver.openai_client import Ds4Backend


def _ok_response(content: str = '{"ok": true}') -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "secret chain of thought",
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_generate_builds_openai_payload(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, timeout_s):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout_s"] = timeout_s
        return _ok_response()

    monkeypatch.setattr(openai_client, "_post_json", fake_post)
    backend = Ds4Backend("http://localhost:8000/")
    resp = backend.generate(
        prompt="hello",
        model="deepseek-v4-flash",
        timeout_s=180.0,
        response_format="json",
        think=False,
        keep_alive="5m",
        options={"temperature": 0, "num_predict": 400},
    )

    assert resp.success
    assert resp.text == '{"ok": true}'
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    p = captured["payload"]
    assert p["model"] == "deepseek-v4-flash"
    assert p["messages"] == [{"role": "user", "content": "hello"}]
    assert p["max_tokens"] == 1500          # num_predict ignored, fixed budget
    assert p["reasoning_effort"] == "low"   # think=False mapping
    assert p["temperature"] == 0
    assert "response_format" not in p       # server ignores it; not sent
    assert "keep_alive" not in p            # Ollama-specific; not sent
    assert "num_predict" not in p
    assert captured["timeout_s"] == 180.0


def test_generate_ignores_reasoning_content(monkeypatch):
    monkeypatch.setattr(openai_client, "_post_json", lambda *a, **k: _ok_response("answer"))
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert resp.text == "answer"
    assert "secret" not in resp.text


def test_generate_rejects_images_without_http_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("HTTP must not be called for images")

    monkeypatch.setattr(openai_client, "_post_json", boom)
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m", images_b64=["Zm9v"])
    assert not resp.success
    assert "vision" in (resp.error or "")


def test_generate_maps_http_error_to_llmresponse(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(openai_client, "_post_json", boom)
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert not resp.success
    assert "TimeoutError" in (resp.error or "")


def test_generate_empty_content_is_error(monkeypatch):
    monkeypatch.setattr(openai_client, "_post_json", lambda *a, **k: _ok_response("   "))
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert not resp.success
    assert "empty" in (resp.error or "")


def test_generate_error_body_is_error(monkeypatch):
    monkeypatch.setattr(
        openai_client, "_post_json",
        lambda *a, **k: {"error": {"message": "model not found", "type": "invalid_request_error"}},
    )
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert not resp.success
    assert "model not found" in (resp.error or "")


def test_generate_malformed_response_is_error(monkeypatch):
    monkeypatch.setattr(openai_client, "_post_json", lambda *a, **k: {"choices": []})
    resp = Ds4Backend("http://localhost:8000").generate(prompt="q", model="m")
    assert not resp.success
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_openai_client.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'archiver.openai_client'`
(run from the repo root so `archiver` is importable)

- [ ] **Step 3: Write the implementation**

Create `archiver/openai_client.py`:

```python
"""OpenAI-compatible chat-completions backend ("ds4").

Implements the LLMBackend protocol over POST /v1/chat/completions.
Targets local reasoning models: reads ONLY message.content (never
reasoning_content), forces low reasoning effort and a fixed max_tokens
budget large enough that the reasoning phase cannot swallow the answer.
"""
from __future__ import annotations

import json
from typing import Any, Optional
from urllib.request import Request, urlopen

from .llm_backend import BaseLLMBackend, LLMResponse

_MAX_TOKENS = 1500


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


class Ds4Backend(BaseLLMBackend):
    """OpenAI-compatible LLM backend (text only).

    Usage:
        backend = Ds4Backend("http://localhost:8000")
        response = backend.generate(prompt="Hello", model="deepseek-v4-flash")
    """

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        timeout_s: float = 120.0,
        images_b64: Optional[list[str]] = None,
        response_format: str | dict[str, Any] | None = None,
        think: bool | str | None = None,
        keep_alive: str | int | None = None,
        options: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        if images_b64:
            return LLMResponse(text="", error="ds4: vision not supported", done=False)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": _MAX_TOKENS,
        }
        if think is False:
            payload["reasoning_effort"] = "low"
        temperature = (options or {}).get("temperature")
        if isinstance(temperature, (int, float)):
            payload["temperature"] = temperature
        # response_format is accepted but not enforced by the server; keep_alive
        # and num_predict are Ollama-specific. None of them are sent: JSON shape
        # is guaranteed by the prompts + the existing normalizer/JSON repair.

        try:
            data = _post_json(f"{self.base_url}/v1/chat/completions", payload, timeout_s=timeout_s)
        except Exception as exc:
            return LLMResponse(text="", error=f"{type(exc).__name__}: {exc}", done=False)

        err = data.get("error")
        if err:
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return LLMResponse(text="", error=str(msg), done=False)

        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return LLMResponse(text="", error="ds4: malformed response", done=False)
        if not content.strip():
            return LLMResponse(text="", error="ds4: empty content", done=False)
        return LLMResponse(text=content, model=data.get("model") or model, done=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_openai_client.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add archiver/openai_client.py tests/test_openai_client.py scripts/check_no_private_host.sh
git commit -m "feat: add ds4 openai-compatible backend client"
```

---

### Task 2: Router (`archiver/llm_router.py`)

**Files:**
- Create: `archiver/llm_router.py`
- Test: `tests/test_llm_router.py`

**Interfaces:**
- Consumes: `Ds4Backend` (Task 1); `OllamaGenerateResult`, `generate`, `generate_with_image_file` from `archiver/ollama_client.py` (existing).
- Produces (used by Task 3):
  - `DS4_PREFIX: str = "ds4:"`
  - `is_ds4_model(model: str) -> bool`
  - `generate(*, model, prompt, base_url="http://localhost:11434", ds4_base_url="", timeout_s=120.0, images_b64=None, response_format=None, think=None, keep_alive=None, options=None) -> OllamaGenerateResult`
  - `generate_with_image_file(*, model, prompt, image_path, base_url="http://localhost:11434", ds4_base_url="", timeout_s=180.0) -> OllamaGenerateResult`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_router.py`:

```python
from __future__ import annotations

from archiver import llm_router
from archiver.llm_backend import LLMResponse
from archiver.ollama_client import OllamaGenerateResult


def test_prefix_constant_and_predicate():
    assert llm_router.DS4_PREFIX == "ds4:"
    assert llm_router.is_ds4_model("ds4:deepseek-v4-flash")
    assert not llm_router.is_ds4_model("gemma3:1b")


def test_ds4_model_routes_to_ds4_backend(monkeypatch):
    captured = {}

    class FakeBackend:
        def __init__(self, base_url):
            captured["base_url"] = base_url

        def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return LLMResponse(text="hi", model="deepseek-v4-flash")

    monkeypatch.setattr(llm_router, "Ds4Backend", FakeBackend)

    def boom(**kwargs):
        raise AssertionError("ollama must not be called for ds4 models")

    monkeypatch.setattr(llm_router, "_ollama_generate", boom)

    res = llm_router.generate(
        model="ds4:deepseek-v4-flash",
        prompt="q",
        ds4_base_url="http://localhost:8000",
        think=False,
        options={"temperature": 0},
    )
    assert isinstance(res, OllamaGenerateResult)
    assert res.error is None
    assert res.response == "hi"
    assert res.model == "ds4:deepseek-v4-flash"  # keeps the prefixed id for cache/UI
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["kwargs"]["model"] == "deepseek-v4-flash"  # prefix stripped


def test_plain_model_routes_to_ollama(monkeypatch):
    captured = {}

    def fake_ollama(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response="ok", model="gemma3:1b", done=True)

    monkeypatch.setattr(llm_router, "_ollama_generate", fake_ollama)
    res = llm_router.generate(
        model="gemma3:1b",
        prompt="q",
        base_url="http://localhost:11434",
        ds4_base_url="http://localhost:8000",
    )
    assert res.response == "ok"
    assert captured["model"] == "gemma3:1b"
    assert "ds4_base_url" not in captured  # ollama API unchanged


def test_ds4_without_endpoint_is_error(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("no backend must be called")

    monkeypatch.setattr(llm_router, "_ollama_generate", boom)
    res = llm_router.generate(model="ds4:deepseek-v4-flash", prompt="q", ds4_base_url="")
    assert res.error
    assert "not configured" in res.error


def test_image_file_on_ds4_is_error(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    res = llm_router.generate_with_image_file(
        model="ds4:deepseek-v4-flash",
        prompt="q",
        image_path=str(img),
        ds4_base_url="http://localhost:8000",
    )
    assert res.error
    assert "vision" in res.error


def test_image_file_on_ollama_delegates(monkeypatch, tmp_path):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response="desc", model="moondream:latest", done=True)

    monkeypatch.setattr(llm_router, "_ollama_generate_with_image_file", fake)
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    res = llm_router.generate_with_image_file(
        model="moondream:latest", prompt="q", image_path=str(img), ds4_base_url="http://localhost:8000"
    )
    assert res.response == "desc"
    assert captured["model"] == "moondream:latest"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_llm_router.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'archiver.llm_router'`

- [ ] **Step 3: Write the implementation**

Create `archiver/llm_router.py`:

```python
"""Route LLM calls to the right backend based on the model-id prefix.

Convention: model ids starting with "ds4:" go to the OpenAI-compatible
endpoint configured via Settings.ds4_base_url (prefix stripped before the
HTTP call); every other model id goes to Ollama. The prefix travels with
the model id everywhere (candidates, settings, cache model_used, UI), so
no extra state is needed to know a model's provider.
"""
from __future__ import annotations

from typing import Any, Optional

from .ollama_client import OllamaGenerateResult
from .ollama_client import generate as _ollama_generate
from .ollama_client import generate_with_image_file as _ollama_generate_with_image_file
from .openai_client import Ds4Backend

DS4_PREFIX = "ds4:"


def is_ds4_model(model: str) -> bool:
    return model.startswith(DS4_PREFIX)


def generate(
    *,
    model: str,
    prompt: str,
    base_url: str = "http://localhost:11434",
    ds4_base_url: str = "",
    timeout_s: float = 120.0,
    images_b64: Optional[list[str]] = None,
    response_format: str | dict[str, Any] | None = None,
    think: bool | str | None = None,
    keep_alive: str | int | None = None,
    options: Optional[dict[str, Any]] = None,
) -> OllamaGenerateResult:
    if is_ds4_model(model):
        if not ds4_base_url:
            return OllamaGenerateResult(response="", error="ds4 endpoint not configured", done=False)
        resp = Ds4Backend(ds4_base_url).generate(
            prompt=prompt,
            model=model[len(DS4_PREFIX):],
            timeout_s=timeout_s,
            images_b64=images_b64,
            response_format=response_format,
            think=think,
            keep_alive=keep_alive,
            options=options,
        )
        return OllamaGenerateResult(response=resp.text, model=model, done=resp.done, error=resp.error)
    return _ollama_generate(
        model=model,
        prompt=prompt,
        base_url=base_url,
        timeout_s=timeout_s,
        images_b64=images_b64,
        response_format=response_format,
        think=think,
        keep_alive=keep_alive,
        options=options,
    )


def generate_with_image_file(
    *,
    model: str,
    prompt: str,
    image_path: str,
    base_url: str = "http://localhost:11434",
    ds4_base_url: str = "",
    timeout_s: float = 180.0,
) -> OllamaGenerateResult:
    if is_ds4_model(model):
        return OllamaGenerateResult(response="", error="ds4: vision not supported", done=False)
    return _ollama_generate_with_image_file(
        model=model,
        prompt=prompt,
        image_path=image_path,
        base_url=base_url,
        timeout_s=timeout_s,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_llm_router.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add archiver/llm_router.py tests/test_llm_router.py
git commit -m "feat: add llm router dispatching by ds4: model prefix"
```

---

### Task 3: Thread `ds4_base_url` through `analyzer.py`

**Files:**
- Modify: `archiver/analyzer.py` (import at line 10; `AnalysisConfig` ~line 46; `_repair_json_dict_via_llm` ~line 222; `_classify_from_text` ~line 288; `_try_text_models` ~line 472; `_extract_facts_from_text` ~line 501; `extract_facts_item` call sites ~lines 631 and 683)
- Test: `tests/test_analyzer_routing.py`

**Interfaces:**
- Consumes: `llm_router.generate` (Task 2 — signature above).
- Produces (used by Tasks 5-6): `AnalysisConfig.ds4_base_url: str = ""` (new frozen field). Internal functions gain keyword-only `ds4_base_url: str = ""` params; public entry points `extract_facts_item(item, *, config)` / classify flow are unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analyzer_routing.py`:

```python
from __future__ import annotations

import json

from archiver import analyzer
from archiver.analyzer import AnalysisConfig, _classify_from_text, _extract_facts_from_text
from archiver.ollama_client import OllamaGenerateResult
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)


def test_analysis_config_has_ds4_base_url_default_empty():
    assert AnalysisConfig().ds4_base_url == ""


def _fake_generate(captured, payload_text):
    def fake(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response=payload_text, model=kwargs["model"], done=True)

    return fake


def test_classify_passes_ds4_base_url_to_generate(monkeypatch):
    captured = {}
    out = json.dumps({"category": "unknown", "reference_year": None, "proposed_name": "doc"})
    monkeypatch.setattr(analyzer, "generate", _fake_generate(captured, out))

    _classify_from_text(
        model="ds4:deepseek-v4-flash",
        content="some text",
        filename="a.pdf",
        mtime_iso="2026-01-01T00:00:00",
        base_url="http://localhost:11434",
        ds4_base_url="http://localhost:8000",
        reference_year_hint=None,
        category_hint=None,
        output_language="en",
        taxonomy=_TAXONOMY,
        filename_separator="space",
    )
    assert captured["ds4_base_url"] == "http://localhost:8000"
    assert captured["model"] == "ds4:deepseek-v4-flash"


def test_facts_passes_ds4_base_url_to_generate(monkeypatch):
    captured = {}
    out = json.dumps({"summary_long": "A letter about something.", "confidence": 0.9})
    monkeypatch.setattr(analyzer, "generate", _fake_generate(captured, out))

    res = _extract_facts_from_text(
        model="ds4:deepseek-v4-flash",
        content="some text",
        filename="a.pdf",
        mtime_iso="2026-01-01T00:00:00",
        base_url="http://localhost:11434",
        ds4_base_url="http://localhost:8000",
        year_hint_filename=None,
        year_hint_text=None,
        output_language="en",
    )
    assert captured["ds4_base_url"] == "http://localhost:8000"
    assert res.status == "scanned"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_analyzer_routing.py -v`
Expected: FAIL — `AnalysisConfig` has no `ds4_base_url`; `_classify_from_text` got unexpected keyword `ds4_base_url`

- [ ] **Step 3: Modify `archiver/analyzer.py`**

3a. Line 10, replace the import:

```python
from .llm_router import generate
```

3b. Add the field to `AnalysisConfig` (after `ollama_base_url`):

```python
    ollama_base_url: str = "http://localhost:11434"
    ds4_base_url: str = ""
```

3c. `_repair_json_dict_via_llm` — new kwarg and pass-through:

```python
def _repair_json_dict_via_llm(*, model: str, raw_output: str, base_url: str, ds4_base_url: str = "") -> Optional[str]:
```

and inside its `generate(...)` call add `ds4_base_url=ds4_base_url,` after `base_url=base_url,`.

3d. `_classify_from_text` — add keyword param `ds4_base_url: str = ""` (after `base_url: str`); in its `generate(...)` call add `ds4_base_url=ds4_base_url,`; in its repair call:

```python
        repaired = _repair_json_dict_via_llm(model=model, raw_output=out, base_url=base_url, ds4_base_url=ds4_base_url)
```

Also update the two error strings in this function from `f"Ollama errore: ..."` to `f"LLM errore: ..."` (the model may not be Ollama anymore).

3e. `_extract_facts_from_text` — same three changes as 3d (param, generate call, repair call, error strings).

3f. `_try_text_models` — pass the config value:

```python
        res = _classify_from_text(
            model=model,
            content=content,
            filename=filename,
            mtime_iso=mtime_iso,
            base_url=cfg.ollama_base_url,
            ds4_base_url=cfg.ds4_base_url,
            ...
```

3g. `extract_facts_item` — both `_extract_facts_from_text(...)` call sites (text kinds ~line 631, image kind ~line 683) add `ds4_base_url=config.ds4_base_url,` after `base_url=config.ollama_base_url,`. The image/vision path (`extract_image_smart`) is intentionally untouched.

- [ ] **Step 4: Run the full suite to verify pass + no regressions**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add archiver/analyzer.py tests/test_analyzer_routing.py
git commit -m "feat: route analyzer llm calls through llm_router with ds4 endpoint"
```

---

### Task 4: Discovery + model selection

**Files:**
- Modify: `archiver/discovery.py` (add `_get_json`, `_discover_ds4`; change `discover_providers` signature)
- Modify: `archiver/model_selection.py` (ds4 preference head; provider merge; vision exclusion)
- Test: `tests/test_discovery_ds4.py`, `tests/test_model_selection_ds4.py`

**Interfaces:**
- Consumes: `DS4_PREFIX` from `llm_router` (Task 2); `ProviderInfo`, `DiscoveryResult` (existing).
- Produces (used by Task 6): `discover_providers(*, ds4_base_url: str = "") -> DiscoveryResult`; ds4 models appear as `ds4:<id>` in `ProviderInfo(name="ds4").models`; `pick_model_candidates` returns ds4 text models first and never in vision.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_discovery_ds4.py`:

```python
from __future__ import annotations

from archiver import discovery
from archiver.discovery import DiscoveryResult, ProviderInfo, _discover_ds4, discover_providers


def test_discover_ds4_prefixes_model_ids(monkeypatch):
    monkeypatch.setattr(
        discovery, "_get_json",
        lambda url, *, timeout_s: {"object": "list", "data": [
            {"id": "deepseek-v4-flash", "object": "model"},
            {"id": "deepseek-v4-pro", "object": "model"},
        ]},
    )
    info = _discover_ds4("http://localhost:8000")
    assert info.name == "ds4"
    assert info.available
    assert info.models == ("ds4:deepseek-v4-flash", "ds4:deepseek-v4-pro")


def test_discover_ds4_server_down(monkeypatch):
    def boom(url, *, timeout_s):
        raise OSError("connection refused")

    monkeypatch.setattr(discovery, "_get_json", boom)
    info = _discover_ds4("http://localhost:8000")
    assert info.name == "ds4"
    assert not info.available
    assert info.models == ()


def test_discover_providers_skips_ds4_when_url_empty(monkeypatch):
    monkeypatch.setattr(
        discovery, "_discover_ollama",
        lambda: ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",)),
    )

    def boom(base_url):
        raise AssertionError("ds4 must not be probed when url is empty")

    monkeypatch.setattr(discovery, "_discover_ds4", boom)
    result = discover_providers(ds4_base_url="")
    assert [p.name for p in result.providers] == ["ollama"]


def test_discover_providers_includes_ds4(monkeypatch):
    monkeypatch.setattr(
        discovery, "_discover_ollama",
        lambda: ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",)),
    )
    monkeypatch.setattr(
        discovery, "_discover_ds4",
        lambda base_url: ProviderInfo(name="ds4", available=True, details="OK", models=("ds4:deepseek-v4-flash",)),
    )
    result = discover_providers(ds4_base_url="http://localhost:8000")
    assert [p.name for p in result.providers] == ["ollama", "ds4"]
```

Create `tests/test_model_selection_ds4.py`:

```python
from __future__ import annotations

from archiver.discovery import DiscoveryResult, ProviderInfo
from archiver.model_selection import pick_model_candidates


def _discovery(ollama_models=(), ds4_models=()):
    providers = []
    if ollama_models:
        providers.append(ProviderInfo(name="ollama", available=True, details="OK", models=tuple(ollama_models)))
    if ds4_models:
        providers.append(ProviderInfo(name="ds4", available=True, details="OK", models=tuple(ds4_models)))
    return DiscoveryResult(providers=tuple(providers))


def test_ds4_models_lead_text_candidates():
    text, vision = pick_model_candidates(
        _discovery(
            ollama_models=("gemma3:1b", "moondream:latest", "qwen2.5:3b-instruct"),
            ds4_models=("ds4:deepseek-v4-flash", "ds4:deepseek-v4-pro"),
        )
    )
    assert text[0] == "ds4:deepseek-v4-flash"
    assert text[1] == "ds4:deepseek-v4-pro"
    assert "gemma3:1b" in text


def test_ds4_models_never_in_vision():
    text, vision = pick_model_candidates(
        _discovery(
            ollama_models=("moondream:latest", "llava:7b"),
            ds4_models=("ds4:deepseek-v4-flash",),
        )
    )
    assert all(not m.startswith("ds4:") for m in vision)
    assert "moondream:latest" in vision


def test_ollama_only_unchanged():
    text, vision = pick_model_candidates(_discovery(ollama_models=("gemma3:1b", "qwen2.5:3b-instruct")))
    assert text[0] == "gemma3:1b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_discovery_ds4.py tests/test_model_selection_ds4.py -v`
Expected: FAIL — `_discover_ds4`/`_get_json` don't exist; `discover_providers()` takes no `ds4_base_url`; ds4 models missing from candidates

- [ ] **Step 3: Modify `archiver/discovery.py`**

Add after the imports (`json` must be added to the import block):

```python
import json
from urllib.request import urlopen


def _get_json(url: str, *, timeout_s: float) -> dict:
    with urlopen(url, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _discover_ds4(base_url: str) -> ProviderInfo:
    from .llm_router import DS4_PREFIX

    url = base_url.rstrip("/") + "/v1/models"
    try:
        data = _get_json(url, timeout_s=2.5)
    except Exception as exc:
        return ProviderInfo(name="ds4", available=False, details=f"Not reachable ({type(exc).__name__})")

    models: list[str] = []
    entries = data.get("data") if isinstance(data, dict) else None
    for entry in entries or []:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and model_id.strip():
            models.append(DS4_PREFIX + model_id.strip())

    details = "OK" if models else "OK (no models listed)"
    return ProviderInfo(name="ds4", available=True, details=details, models=tuple(models))
```

Change `discover_providers` to:

```python
def discover_providers(*, ds4_base_url: str = "") -> DiscoveryResult:
    providers: list[ProviderInfo] = []
    notes: list[str] = []

    ollama = _discover_ollama()
    providers.append(ollama)

    ds4 = None
    if ds4_base_url.strip():
        ds4 = _discover_ds4(ds4_base_url.strip())
        providers.append(ds4)

    chosen_text = None
    chosen_vision = None
    if ds4 is not None and ds4.available and ds4.models:
        chosen_text = "ds4"
    elif ollama.available and ollama.models:
        chosen_text = "ollama"
    elif ollama.available and not ollama.models:
        notes.append("Ollama is installed but has no models: run 'ollama pull <model>'.")

    return DiscoveryResult(
        providers=tuple(providers),
        chosen_text=chosen_text,
        chosen_vision=chosen_vision,
        notes=tuple(notes),
    )
```

- [ ] **Step 4: Modify `archiver/model_selection.py`**

4a. Add at the top (after existing imports):

```python
from .llm_router import DS4_PREFIX
```

4b. Add the ds4 preference head and prepend it to `_TEXT_PREFER`:

```python
_DS4_TEXT_PREFER = (
    "ds4:deepseek-v4-flash",
    "ds4:deepseek-v4-pro",
)

_TEXT_PREFER = _DS4_TEXT_PREFER + (
    "gemma3:1b",
    "qwen2.5:3b-instruct",
    "phi4-mini:latest",
    "phi4-mini",
    "qwen3:4b",
    "qwen3.5:4b",
    "ministral-3:3b",
    "gemma2:2b",
    "qwen2.5:7b",
    "mistral:latest",
    "gemma3:latest",
)
```

(the eleven ollama entries are the current `_TEXT_PREFER` literal, unchanged)

4c. In `pick_model_candidates`, replace the provider loop:

```python
    models: list[str] = []
    if discovery:
        for p in discovery.providers:
            if p.name in ("ollama", "ds4") and p.available and p.models:
                models.extend(p.models)
```

4d. Exclude ds4 from vision candidates:

```python
    vision_candidates = [model for model in models if _is_vision_model(model) and not model.startswith(DS4_PREFIX)]
```

- [ ] **Step 5: Run the full suite**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: all PASS (check `test_ollama_only_unchanged` in particular — the ds4 entries in `_TEXT_PREFER` must not disturb ollama-only ordering)

- [ ] **Step 6: Commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add archiver/discovery.py archiver/model_selection.py tests/test_discovery_ds4.py tests/test_model_selection_ds4.py
git commit -m "feat: discover ds4 provider and rank its models first for text"
```

---

### Task 5: Config threading (`config.py`, `settings.py`, `setup_logic.py`, `__main__.py`, `task_builders.py`)

**Files:**
- Modify: `archiver/config.py` (AppConfig field + `load_config` parsing)
- Modify: `archiver/settings.py` (Settings field)
- Modify: `archiver/setup_logic.py` (both converters)
- Modify: `archiver/__main__.py` (Settings construction ~line 76)
- Modify: `archiver/task_builders.py` (prefer_fast head + AnalysisConfig threading)
- Test: `tests/test_config_threading.py`

**Interfaces:**
- Consumes: `AnalysisConfig.ds4_base_url` (Task 3).
- Produces (used by Task 6): `AppConfig.ds4_base_url: str = ""`, `Settings.ds4_base_url: str = ""`; `build_analysis_config()` emits `AnalysisConfig` with `ds4_base_url` set and ds4-first facts ordering.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_threading.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from archiver.config import AppConfig, load_config, save_config
from archiver.discovery import DiscoveryResult, ProviderInfo
from archiver.settings import Settings
from archiver.setup_logic import app_config_from_settings
from archiver.task_builders import build_analysis_config
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)


def test_config_roundtrip_ds4_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_config(AppConfig(ds4_base_url="http://localhost:8000"))
    written = json.loads((tmp_path / "amenity-stuff" / "config.json").read_text())
    assert written["ds4_base_url"] == "http://localhost:8000"
    assert load_config().ds4_base_url == "http://localhost:8000"


def test_config_default_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert load_config().ds4_base_url == ""


def test_app_config_from_settings_carries_ds4():
    settings = Settings(
        source_root=Path("."),
        archive_root=Path("./ARCHIVE"),
        ds4_base_url="http://localhost:8000",
    )
    assert app_config_from_settings(settings).ds4_base_url == "http://localhost:8000"


def test_build_analysis_config_threads_ds4_and_prefers_flash():
    settings = Settings(
        source_root=Path("."),
        archive_root=Path("./ARCHIVE"),
        ds4_base_url="http://localhost:8000",
    )
    discovery = DiscoveryResult(
        providers=(
            ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",)),
            ProviderInfo(name="ds4", available=True, details="OK",
                         models=("ds4:deepseek-v4-flash", "ds4:deepseek-v4-pro")),
        )
    )
    cfg = build_analysis_config(settings=settings, discovery=discovery, taxonomy=_TAXONOMY)
    assert cfg.ds4_base_url == "http://localhost:8000"
    assert cfg.text_models[0] == "ds4:deepseek-v4-flash"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_config_threading.py -v`
Expected: FAIL — unexpected keyword `ds4_base_url` on `AppConfig`/`Settings`; missing on `AnalysisConfig` output

- [ ] **Step 3: Apply the modifications**

3a. `archiver/config.py` — add field after `undated_folder_name`:

```python
    undated_folder_name: str = "undated"
    ds4_base_url: str = ""  # OpenAI-compatible endpoint; empty = disabled
```

In `load_config()`, add with the other `data.get(...)` reads:

```python
    ds4_base_url = data.get("ds4_base_url")
```

and with the other parsers (before `return AppConfig(**kwargs)`):

```python
    if isinstance(ds4_base_url, str) and ds4_base_url.strip():
        kwargs["ds4_base_url"] = ds4_base_url.strip()
```

(`save_config` needs no change: it serializes `__dict__`.)

3b. `archiver/settings.py` — add field after `undated_folder_name`:

```python
    undated_folder_name: str = "undated"
    ds4_base_url: str = ""  # OpenAI-compatible endpoint; empty = disabled
```

3c. `archiver/setup_logic.py` — add to BOTH constructors:

```python
        undated_folder_name=current.undated_folder_name,
        ds4_base_url=current.ds4_base_url,
```

in `settings_from_setup`, and

```python
        undated_folder_name=settings.undated_folder_name,
        ds4_base_url=settings.ds4_base_url,
```

in `app_config_from_settings`.

3d. `archiver/__main__.py` — in the `Settings(...)` construction add:

```python
        undated_folder_name=cfg.undated_folder_name,
        ds4_base_url=cfg.ds4_base_url,
```

3e. `archiver/task_builders.py` — inside `build_analysis_config`, change `prefer_fast` to start with the ds4 models:

```python
        prefer_fast = (
            "ds4:deepseek-v4-flash",
            "ds4:deepseek-v4-pro",
            "gemma3:1b",
            "qwen2.5:3b-instruct",
            "phi4-mini:latest",
            "phi4-mini",
            "qwen3:4b",
            "qwen3.5:4b",
            "ministral-3:3b",
            "gemma2:2b",
            "qwen2.5:7b",
        )
```

and add the field to the returned config:

```python
    return AnalysisConfig(
        output_language=settings.output_language,
        taxonomy=taxonomy,
        text_models=text_models,
        vision_models=vision_models,
        filename_separator=settings.filename_separator,
        ocr_mode=settings.ocr_mode,
        ds4_base_url=settings.ds4_base_url,
    )
```

- [ ] **Step 4: Run the full suite**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add archiver/config.py archiver/settings.py archiver/setup_logic.py archiver/__main__.py archiver/task_builders.py tests/test_config_threading.py
git commit -m "feat: persist and thread ds4_base_url through config and task builders"
```

---

### Task 6: TUI wiring (`app.py`, `settings_screen.py`, `ui_status.py`)

**Files:**
- Modify: `archiver/app.py` (`_ordered_classify_models` ~line 141; `action_settings` ~line 314; `_on_settings_done` ~line 355; `do_discover` ~line 392)
- Modify: `archiver/settings_screen.py` (`SettingsResult`, `__init__`, `compose`, `_save`, `action_cancel`, `_filter_vision_models`)
- Modify: `archiver/ui_status.py` (`provider_summary`)
- Test: `tests/test_settings_screen_result.py` (dataclass smoke; widget behavior is verified manually in Task 7)

**Interfaces:**
- Consumes: `discover_providers(ds4_base_url=...)` (Task 4); `Settings.ds4_base_url` (Task 5).
- Produces: `SettingsResult.ds4_base_url: str` (new frozen field); `SettingsScreen.__init__` gains keyword-only `ds4_base_url: str` parameter.

- [ ] **Step 1: Write the failing test**

Create `tests/test_settings_screen_result.py`:

```python
from __future__ import annotations

from pathlib import Path

from archiver.settings_screen import SettingsResult


def test_settings_result_carries_ds4_base_url():
    r = SettingsResult(
        output_language="auto",
        taxonomies={},
        facts_model="auto",
        classify_model="auto",
        vision_model="auto",
        vision_model_fallback="none",
        filename_separator="space",
        ocr_mode="balanced",
        undated_folder_name="undated",
        archive_root=Path("./ARCHIVE"),
        ds4_base_url="http://localhost:8000",
    )
    assert r.ds4_base_url == "http://localhost:8000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_settings_screen_result.py -v`
Expected: FAIL with unexpected keyword `ds4_base_url`

- [ ] **Step 3: Modify `archiver/settings_screen.py`**

3a. Add `Input` to the textual imports:

```python
from textual.widgets import Footer, Header, Input, OptionList, Static, TextArea
```

3b. `SettingsResult` — add field:

```python
    archive_root: Path
    ds4_base_url: str
```

3c. `__init__` — add keyword param `ds4_base_url: str,` (after `archive_root: Path,`) and store:

```python
        self._ds4_base_url = (ds4_base_url or "").strip()
```

3d. `compose()` — after the `#provider` Static, add:

```python
        yield Static("ds4 endpoint (OpenAI-compatible, empty = disabled):", id="ds4_label")
        yield Input(value=self._ds4_base_url, placeholder="http://localhost:8000", id="ds4_url")
```

and add CSS rules inside the existing `CSS` string:

```
    #ds4_label { height: auto; padding: 1 0 0 0; }
    #ds4_url { height: 3; }
```

3e. Add a helper and use it in BOTH dismiss paths (`action_cancel` and `_save`), adding `ds4_base_url=self._current_ds4_url(),` to each `SettingsResult(...)` construction:

```python
    def _current_ds4_url(self) -> str:
        try:
            return self.query_one("#ds4_url", Input).value.strip()
        except Exception:
            return self._ds4_base_url
```

3f. `_filter_vision_models` — first line of the loop body, add the guard:

```python
        for m in models:
            ml = m.lower()
            if ml.startswith("ds4:"):
                continue
```

(`_filter_text_models` needs no change: ds4 ids contain no vision tokens, so they already pass.)

- [ ] **Step 4: Modify `archiver/app.py`**

4a. `do_discover` (inside `_run_discovery`):

```python
        def do_discover() -> DiscoveryResult:
            return discover_providers(ds4_base_url=self.settings.ds4_base_url)
```

4b. `_ordered_classify_models` — prepend to `prefer`:

```python
        prefer = (
            "ds4:deepseek-v4-flash",
            "ds4:deepseek-v4-pro",
            "qwen2.5:3b-instruct",
            "qwen3:4b",
            "phi4-mini:latest",
            "phi4-mini",
            "gemma3:1b",
            "qwen3.5:4b",
            "ministral-3:3b",
            "gemma2:2b",
        )
```

4c. `action_settings` — replace the ollama-only model collection loop with a merge over all available providers:

```python
        if self._discovery:
            provider_info = self._provider_line or "Provider: (not detected)"
            merged: list[str] = []
            for p in self._discovery.providers:
                if p.available and p.models:
                    merged.extend(p.models)
            available_models = tuple(merged)
```

and pass the endpoint to the screen (after `archive_root=...`):

```python
                archive_root=self.settings.archive_root,
                ds4_base_url=self.settings.ds4_base_url,
```

4d. `_on_settings_done` — capture the old value, apply the new one, re-discover if it changed:

```python
    def _on_settings_done(self, result: SettingsResult) -> None:
        ds4_changed = result.ds4_base_url != self.settings.ds4_base_url
        self.settings = replace(
            self.settings,
            output_language=result.output_language,
            taxonomies=result.taxonomies,
            facts_model=result.facts_model,
            classify_model=result.classify_model,
            vision_model=result.vision_model,
            vision_model_fallback=result.vision_model_fallback,
            archive_root=result.archive_root,
            filename_separator=result.filename_separator,
            ocr_mode=result.ocr_mode,
            undated_folder_name=result.undated_folder_name,
            ds4_base_url=result.ds4_base_url,
        )
        self.query_one("#arc", Static).update(f"Archive: {self.settings.archive_root}")
        self._save_app_config()
        self._render_notes()
        if ds4_changed:
            self.run_worker(self._run_discovery())
```

- [ ] **Step 5: Modify `archiver/ui_status.py`**

Replace the provider-gathering block at the top of `provider_summary` (keep everything from `text_models, vision_models = model_picker(discovery)` down unchanged):

```python
    names: list[str] = []
    models: tuple[str, ...] = ()
    for p in discovery.providers:
        if p.available:
            names.append(p.name)
            models = models + p.models
        elif p.name == "ollama":
            names.append("ollama(missing)")
    if not names:
        return ""
    provider = "+".join(names)
```

- [ ] **Step 6: Run the full suite + import smoke check**

```bash
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v
~/.local/share/amenity-stuff/venv/bin/python -c "import archiver.app, archiver.settings_screen, archiver.ui_status"
```

Expected: all PASS; imports clean.

- [ ] **Step 7: Commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add archiver/app.py archiver/settings_screen.py archiver/ui_status.py tests/test_settings_screen_result.py
git commit -m "feat: expose ds4 endpoint in settings ui and provider summary"
```

---

### Task 7: Docs, version bump, manual PTY verification

**Files:**
- Modify: `README.md` (LLM Provider + Settings sections), `CLAUDE.md` (Model Selection section), `AGENTS.md` (routing convention note)
- Modify: `VERSION` (`0.9.13` → `0.10.0`), `pyproject.toml` (same), `CHANGELOG.md`

**Interfaces:**
- Consumes: everything above, complete.
- Produces: released docs + version; verified app.

- [ ] **Step 1: Update README.md**

In the "LLM Provider" section add:

```markdown
### Additional OpenAI-compatible endpoint (ds4)

Besides Ollama, you can point amenity-ai at any local OpenAI-compatible server
(e.g. vLLM, llama.cpp server). Set the endpoint in Settings (`F2`) →
"ds4 endpoint", e.g. `http://localhost:8000`. Leave it empty to disable.

Models from that server appear with the `ds4:` prefix (e.g. `ds4:deepseek-v4-flash`)
and are preferred for facts/classify when models are set to `auto`. They are
text-only: images always use Ollama vision models.
```

In the "Settings" list add: `- ds4 endpoint (OpenAI-compatible server, optional)`.

- [ ] **Step 2: Update CLAUDE.md and AGENTS.md**

CLAUDE.md — in "Model Selection (\"auto\")", append:

```markdown
- A second provider ("ds4", any OpenAI-compatible server) is routed by model-id prefix:
  `ds4:<model>` goes through `llm_router.py` → `openai_client.Ds4Backend`; everything else
  goes to Ollama. The endpoint lives in `Settings.ds4_base_url` (empty = disabled) and is
  a user-local setting — never hardcode an endpoint in the repo. ds4 is text-only.
```

AGENTS.md — under "Code Structure" add one bullet:

```markdown
- LLM provider routing is by model-id prefix (`ds4:` → OpenAI-compatible backend via
  `llm_router.py`; no prefix → Ollama). Keep the prefix attached to model ids everywhere
  (settings, candidates, cache `model_used`, UI).
```

- [ ] **Step 3: Version bump + CHANGELOG**

- `VERSION`: `0.10.0`
- `pyproject.toml`: `version = "0.10.0"`
- `CHANGELOG.md`, new entry at top:

```markdown
## 0.10.0

- Add support for a second local LLM provider: any OpenAI-compatible
  chat-completions server ("ds4"), configured via Settings → ds4 endpoint.
  Models appear with the `ds4:` prefix, are preferred for facts/classify in
  `auto` mode, and fall back to Ollama when the server is unavailable.
  Text-only: vision stays on Ollama.
```

- [ ] **Step 4: Reinstall and run the automated suite**

```bash
~/.local/share/amenity-stuff/venv/bin/pip install -e .
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 5: Manual PTY verification (per user rule: emulate the real terminal)**

Precondition: the tester configures their real endpoint ONLY in
`~/.config/amenity-stuff/config.json` (`"ds4_base_url": "http://<your-host>:8000"`) —
never in any repo file.

```bash
mkdir -p /tmp/amenity-sample && cp <a few pdf/txt samples> /tmp/amenity-sample/
script -qec "amenity-ai --source /tmp/amenity-sample --archive /tmp/amenity-archive" /dev/null
```

Checklist:
1. Title/provider line shows `ollama+ds4` and the merged model count.
2. `S` (scan): details panel shows `model_used = ds4:deepseek-v4-flash`, valid facts.
3. `C` (classify): classification completes with a ds4 model.
4. Stop the ds4 server, restart the app: provider line shows only ollama; scan falls back to Ollama models without UI errors.
5. `F2`: endpoint field edits + persists (check `config.json`); vision model choices contain no `ds4:*`.
6. `amenity-ai report --source /tmp/amenity-sample` prints timings.

- [ ] **Step 6: Final commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add README.md CLAUDE.md AGENTS.md VERSION pyproject.toml CHANGELOG.md
git commit -m "docs: document ds4 provider; release 0.10.0"
```

---

## Self-Review Notes

- Spec coverage: backend mapping (Task 1), router+prefix (Task 2), analyzer threading incl. JSON repair (Task 3), discovery+selection priority (Task 4), config persistence/threading incl. `prefer_fast` (Task 5), settings UI + vision exclusion + provider summary (Task 6), docs+0.10.0+manual tests (Task 7). Privacy constraint enforced via Global Constraints + per-commit grep.
- The image/vision path (`extract_image_smart`, `extractors/image.py`) is deliberately untouched: it talks to Ollama directly and ds4 is text-only per spec.
- `analyzer.py` error strings change from "Ollama errore" to "LLM errore" (Task 3) — the only user-visible copy change, required for correctness with two providers.
