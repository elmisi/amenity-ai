# Configurable Ollama URL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Ollama endpoint configurable (default `http://localhost:11434`) so Ollama can run on another machine, symmetric to the existing `ds4_base_url`.

**Architecture:** Provider discovery switches from the `ollama` CLI to `GET {url}/api/tags` over HTTP — that switch is what makes a remote host work at all. The URL then threads through the same chain the ds4 endpoint already uses (AppConfig → Settings → AnalysisConfig) and gets its own Input field in the settings screen.

**Tech Stack:** Python 3.10+, stdlib `urllib` (no new dependencies), Textual 8.x TUI, pytest.

**Spec:** `docs/superpowers/specs/2026-08-01-ollama-configurable-url-design.md`

## Global Constraints

- Default is `http://localhost:11434` everywhere (AppConfig, Settings, `discover_providers`) — **never empty**. Ollama is the base provider; the default must reproduce today's behavior exactly for users who configure nothing. (This differs from `ds4_base_url`, whose default is `""` = disabled.)
- Never write a personal hostname anywhere in the repo. Tests/docs use `http://localhost:11434`, or the invented placeholder `http://ollama-box:11434` where a non-local URL is needed.
- Run `sh scripts/check_no_private_host.sh` before EVERY commit — it must print `OK`.
- No new runtime dependencies. pytest stays dev-only (not in `pyproject.toml`).
- Behavior-preserving: no changes to prompts, model ranking, fallback logic, or timeouts. Discovery timeout stays 2.5 s.
- Errors never raise across the TUI: discovery failures return `ProviderInfo(available=False, ...)`.
- Do NOT bump the version per task. One minor bump `0.10.1 → 0.11.0` (VERSION + pyproject.toml by hand) plus CHANGELOG in the final task.
- Commit messages: `type: description`, no Co-Authored-By lines, no session URLs.
- Tests run from the repo root: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v` (36 tests exist today; all must keep passing).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `archiver/discovery.py` | Modify | `_discover_ollama(base_url)` over HTTP; `discover_providers` gains `ollama_base_url` |
| `archiver/ui_status.py` | Modify | `ollama(missing)` → `ollama(down)` (HTTP probe means unreachable, not missing binary) |
| `archiver/config.py` | Modify | `AppConfig.ollama_base_url` + defensive parsing |
| `archiver/settings.py` | Modify | `Settings.ollama_base_url` |
| `archiver/setup_logic.py` | Modify | Both converters carry the field |
| `archiver/__main__.py` | Modify | Wire `cfg.ollama_base_url` into `Settings` |
| `archiver/task_builders.py` | Modify | Pass it into `AnalysisConfig` (field already exists there) |
| `archiver/app.py` | Modify | Discovery arg; classify `base_url`; settings screen wiring; re-discovery |
| `archiver/settings_screen.py` | Modify | Second Input + `SettingsResult.ollama_base_url` |
| `scripts/check_no_private_host.sh` | Modify | Guard both endpoint hostnames |
| `tests/test_discovery_ollama_http.py` | Create | HTTP discovery unit tests |
| `tests/test_discovery_ds4.py` | Modify | Adapt `_discover_ollama` monkeypatch lambdas to the new signature |
| `tests/test_config_threading.py` | Modify | Roundtrip + threading tests for the new field |
| `tests/test_settings_screen_result.py` | Modify | New field in the dataclass smoke test |
| `README.md`, `PROJECT_SPEC.md`, `CLAUDE.md`, `CHANGELOG.md`, `VERSION`, `pyproject.toml` | Modify | Docs + release 0.11.0 |

---

### Task 1: HTTP discovery for Ollama

**Files:**
- Modify: `archiver/discovery.py` (imports lines 6-11; `ProviderInfo` line 14-20; `_run` lines 31-38; `_discover_ollama` lines 67-94; `discover_providers` lines 96-122)
- Modify: `archiver/ui_status.py` (`provider_summary`, the `ollama(missing)` literal ~line 58)
- Test: `tests/test_discovery_ollama_http.py` (create), `tests/test_discovery_ds4.py` (adapt)

**Interfaces:**
- Consumes: `_get_json(url, *, timeout_s) -> dict` (already exists in `discovery.py`, added for ds4); `ProviderInfo`, `DiscoveryResult`.
- Produces (used by Tasks 2-3): `_discover_ollama(base_url: str) -> ProviderInfo`; `discover_providers(*, ollama_base_url: str = "http://localhost:11434", ds4_base_url: str = "") -> DiscoveryResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_discovery_ollama_http.py`:

```python
from __future__ import annotations

from archiver import discovery
from archiver.discovery import ProviderInfo, _discover_ollama, discover_providers


def test_discover_ollama_parses_api_tags(monkeypatch):
    captured = {}

    def fake_get(url, *, timeout_s):
        captured["url"] = url
        captured["timeout_s"] = timeout_s
        return {"models": [{"name": "gemma3:1b"}, {"name": "moondream:latest"}]}

    monkeypatch.setattr(discovery, "_get_json", fake_get)
    info = _discover_ollama("http://localhost:11434/")

    assert info.name == "ollama"
    assert info.available
    assert info.models == ("gemma3:1b", "moondream:latest")
    assert info.details == "OK"
    assert captured["url"] == "http://localhost:11434/api/tags"  # trailing slash normalized
    assert captured["timeout_s"] == 2.5


def test_discover_ollama_uses_configured_remote_url(monkeypatch):
    captured = {}

    def fake_get(url, *, timeout_s):
        captured["url"] = url
        return {"models": []}

    monkeypatch.setattr(discovery, "_get_json", fake_get)
    info = _discover_ollama("http://ollama-box:11434")

    assert captured["url"] == "http://ollama-box:11434/api/tags"
    assert info.available
    assert info.details == "OK (no models listed)"


def test_discover_ollama_server_down(monkeypatch):
    def boom(url, *, timeout_s):
        raise OSError("connection refused")

    monkeypatch.setattr(discovery, "_get_json", boom)
    info = _discover_ollama("http://localhost:11434")

    assert info.name == "ollama"
    assert not info.available
    assert info.models == ()
    assert "Not reachable" in info.details


def test_discover_ollama_malformed_payload(monkeypatch):
    monkeypatch.setattr(discovery, "_get_json", lambda url, *, timeout_s: ["not", "a", "dict"])
    info = _discover_ollama("http://localhost:11434")
    assert info.available
    assert info.models == ()


def test_discover_ollama_skips_unnamed_entries(monkeypatch):
    monkeypatch.setattr(
        discovery, "_get_json",
        lambda url, *, timeout_s: {"models": [{"name": "gemma3:1b"}, {"size": 12}, "junk", {"name": "  "}]},
    )
    info = _discover_ollama("http://localhost:11434")
    assert info.models == ("gemma3:1b",)


def test_discover_providers_passes_ollama_url(monkeypatch):
    captured = {}

    def fake_ollama(base_url):
        captured["base_url"] = base_url
        return ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",))

    monkeypatch.setattr(discovery, "_discover_ollama", fake_ollama)
    result = discover_providers(ollama_base_url="http://ollama-box:11434")

    assert captured["base_url"] == "http://ollama-box:11434"
    assert [p.name for p in result.providers] == ["ollama"]
    assert result.chosen_text == "ollama"


def test_discover_providers_defaults_to_localhost(monkeypatch):
    captured = {}

    def fake_ollama(base_url):
        captured["base_url"] = base_url
        return ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",))

    monkeypatch.setattr(discovery, "_discover_ollama", fake_ollama)
    discover_providers()
    assert captured["base_url"] == "http://localhost:11434"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_discovery_ollama_http.py -v`
Expected: FAIL — `_discover_ollama()` takes 0 positional arguments (it currently shells out to the CLI and takes none); `discover_providers()` got an unexpected keyword argument `ollama_base_url`

- [ ] **Step 3: Rewrite `_discover_ollama` in `archiver/discovery.py`**

Replace the whole function (currently lines 67-94) with:

```python
def _discover_ollama(base_url: str) -> ProviderInfo:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        data = _get_json(url, timeout_s=2.5)
    except Exception as exc:
        return ProviderInfo(name="ollama", available=False, details=f"Not reachable ({type(exc).__name__})")

    models: list[str] = []
    entries = data.get("models") if isinstance(data, dict) else None
    for entry in entries or []:
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, str) and name.strip():
            models.append(name.strip())

    details = "OK" if models else "OK (no models listed)"
    return ProviderInfo(name="ollama", available=True, details=details, models=tuple(models))
```

- [ ] **Step 4: Update `discover_providers` and prune dead code**

Signature and the ollama call:

```python
def discover_providers(
    *,
    ollama_base_url: str = "http://localhost:11434",
    ds4_base_url: str = "",
) -> DiscoveryResult:
    providers: list[ProviderInfo] = []
    notes: list[str] = []

    ollama = _discover_ollama(ollama_base_url.strip() or "http://localhost:11434")
    providers.append(ollama)
```

Everything from `ds4 = None` down stays byte-identical, EXCEPT the note text (the binary is no longer what we detect):

```python
        notes.append("Ollama has no models: run 'ollama pull <model>'.")
```

Then delete now-dead code:
- the `_run` helper (lines 31-38)
- the `import os`, `from pathlib import Path`, `import shutil`, `import subprocess` lines (all four become unused — `os` and `Path` were already unused before this change)
- the `command: Optional[str] = None` field on `ProviderInfo` — it was only ever set by the CLI path and nothing outside `discovery.py` reads it (verified by grep across `archiver/` and `tests/`)

Keep `import json` and `from urllib.request import urlopen` (used by `_get_json`), and keep `from typing import Optional` (still used by `DiscoveryResult`).

- [ ] **Step 5: Adapt the existing ds4 discovery tests**

In `tests/test_discovery_ds4.py`, two monkeypatches use a zero-arg lambda for `_discover_ollama` (lines ~34 and ~48). Both become:

```python
        lambda base_url: ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",)),
```

- [ ] **Step 6: Update the provider label in `archiver/ui_status.py`**

In `provider_summary`, the unavailable-ollama branch:

```python
        elif p.name == "ollama":
            names.append("ollama(down)")
```

(With HTTP discovery, an unavailable Ollama means the server did not answer — "missing" would now be wrong.)

- [ ] **Step 7: Run the full suite**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: all PASS (36 existing + 7 new = 43)

- [ ] **Step 8: Commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add archiver/discovery.py archiver/ui_status.py tests/test_discovery_ollama_http.py tests/test_discovery_ds4.py
git commit -m "feat: discover ollama over http api/tags instead of the cli"
```

---

### Task 2: Thread `ollama_base_url` through config and the analysis pipeline

**Files:**
- Modify: `archiver/config.py` (`AppConfig` ~line 23; `load_config` reads ~line 69 and parsers ~line 124)
- Modify: `archiver/settings.py` (`Settings` ~line 46)
- Modify: `archiver/setup_logic.py` (`settings_from_setup` ~line 24; `app_config_from_settings` ~line 42)
- Modify: `archiver/__main__.py` (`Settings(...)` construction ~line 88)
- Modify: `archiver/task_builders.py` (`AnalysisConfig(...)` ~line 58)
- Modify: `archiver/app.py` (`do_discover` line 400; classify `base_url` lines 602 and 935)
- Test: `tests/test_config_threading.py`

**Interfaces:**
- Consumes: `discover_providers(*, ollama_base_url=..., ds4_base_url=...)` (Task 1).
- Produces (used by Task 3): `AppConfig.ollama_base_url: str = "http://localhost:11434"`, `Settings.ollama_base_url: str = "http://localhost:11434"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_threading.py`:

```python
def test_config_roundtrip_ollama_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_config(AppConfig(ollama_base_url="http://ollama-box:11434"))
    written = json.loads((tmp_path / "amenity-stuff" / "config.json").read_text())
    assert written["ollama_base_url"] == "http://ollama-box:11434"
    assert load_config().ollama_base_url == "http://ollama-box:11434"


def test_config_ollama_default_is_localhost(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert load_config().ollama_base_url == "http://localhost:11434"


def test_config_ollama_blank_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg_dir = tmp_path / "amenity-stuff"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(json.dumps({"ollama_base_url": "   "}))
    assert load_config().ollama_base_url == "http://localhost:11434"


def test_app_config_from_settings_carries_ollama_url():
    settings = Settings(
        source_root=Path("."),
        archive_root=Path("./ARCHIVE"),
        ollama_base_url="http://ollama-box:11434",
    )
    assert app_config_from_settings(settings).ollama_base_url == "http://ollama-box:11434"


def test_build_analysis_config_threads_ollama_url():
    settings = Settings(
        source_root=Path("."),
        archive_root=Path("./ARCHIVE"),
        ollama_base_url="http://ollama-box:11434",
    )
    discovery = DiscoveryResult(
        providers=(ProviderInfo(name="ollama", available=True, details="OK", models=("gemma3:1b",)),)
    )
    cfg = build_analysis_config(settings=settings, discovery=discovery, taxonomy=_TAXONOMY)
    assert cfg.ollama_base_url == "http://ollama-box:11434"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_config_threading.py -v`
Expected: FAIL — `AppConfig`/`Settings` got an unexpected keyword argument `ollama_base_url`

- [ ] **Step 3: Add the config fields**

`archiver/config.py`, after the `ds4_base_url` field:

```python
    ds4_base_url: str = ""  # OpenAI-compatible endpoint; empty = disabled
    ollama_base_url: str = "http://localhost:11434"  # may point at another machine
```

In `load_config()`, with the other `data.get(...)` reads:

```python
    ollama_base_url = data.get("ollama_base_url")
```

and with the other parsers (before `return AppConfig(**kwargs)`):

```python
    if isinstance(ollama_base_url, str) and ollama_base_url.strip():
        kwargs["ollama_base_url"] = ollama_base_url.strip()
```

(`save_config` needs no change — it serializes `__dict__`.)

`archiver/settings.py`, after its `ds4_base_url` field:

```python
    ds4_base_url: str = ""  # OpenAI-compatible endpoint; empty = disabled
    ollama_base_url: str = "http://localhost:11434"  # may point at another machine
```

- [ ] **Step 4: Thread it through the converters and builders**

`archiver/setup_logic.py` — in `settings_from_setup`, next to the existing ds4 line:

```python
        ds4_base_url=current.ds4_base_url,
        ollama_base_url=current.ollama_base_url,
```

and in `app_config_from_settings`:

```python
        ds4_base_url=settings.ds4_base_url,
        ollama_base_url=settings.ollama_base_url,
```

`archiver/__main__.py` — in the `Settings(...)` construction:

```python
        ds4_base_url=cfg.ds4_base_url,
        ollama_base_url=cfg.ollama_base_url,
```

`archiver/task_builders.py` — in the returned `AnalysisConfig(...)` (the field already exists on `AnalysisConfig`, it was simply never set):

```python
        ds4_base_url=settings.ds4_base_url,
        ollama_base_url=settings.ollama_base_url,
```

- [ ] **Step 5: Use the setting in `archiver/app.py`**

5a. `do_discover` (line 400):

```python
            return discover_providers(
                ollama_base_url=self.settings.ollama_base_url,
                ds4_base_url=self.settings.ds4_base_url,
            )
```

5b. Both classify call sites — line 602 (`_run_classify_batch`) and line 935 (`_run_classify_row`) — replace the hardcoded URL:

```python
                    base_url=self.settings.ollama_base_url,
```

(Match each site's existing indentation; the neighbouring `ds4_base_url=` line stays as it is.)

The image/vision path needs no change: `extract_image_smart` already receives `config.ollama_base_url`, which Step 4 now populates.

- [ ] **Step 6: Run the full suite**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: all PASS (43 + 5 new = 48)

- [ ] **Step 7: Commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add archiver/config.py archiver/settings.py archiver/setup_logic.py archiver/__main__.py archiver/task_builders.py archiver/app.py tests/test_config_threading.py
git commit -m "feat: thread configurable ollama endpoint through config and pipeline"
```

---

### Task 3: Settings screen field, re-discovery, and privacy guard

**Files:**
- Modify: `archiver/settings_screen.py` (`SettingsResult` ~line 32; CSS ~line 44; ctor ~line 70/76; `compose` ~line 125; `_current_ds4_url` neighbourhood ~line 148; both dismiss paths ~line 167 and ~line 301)
- Modify: `archiver/app.py` (`action_settings` ~line 347; `_on_settings_done` ~lines 359-378)
- Modify: `scripts/check_no_private_host.sh`
- Test: `tests/test_settings_screen_result.py`

**Interfaces:**
- Consumes: `Settings.ollama_base_url` (Task 2).
- Produces: `SettingsResult.ollama_base_url: str`; `SettingsScreen.__init__` keyword param `ollama_base_url: str`.

- [ ] **Step 1: Write the failing test**

Replace the body of `tests/test_settings_screen_result.py` with the same construction plus the new field:

```python
from __future__ import annotations

from pathlib import Path

from archiver.settings_screen import SettingsResult


def test_settings_result_carries_endpoints():
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
        ollama_base_url="http://ollama-box:11434",
    )
    assert r.ds4_base_url == "http://localhost:8000"
    assert r.ollama_base_url == "http://ollama-box:11434"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_settings_screen_result.py -v`
Expected: FAIL — unexpected keyword argument `ollama_base_url`

- [ ] **Step 3: Modify `archiver/settings_screen.py`**

3a. `SettingsResult` — add after `ds4_base_url`:

```python
    ds4_base_url: str
    ollama_base_url: str
```

3b. CSS — add next to the ds4 rules:

```
    #ollama_label { height: auto; padding: 1 0 0 0; }
    #ollama_url { height: 3; }
```

3c. `__init__` — add the keyword param after `ds4_base_url: str,`:

```python
        ds4_base_url: str,
        ollama_base_url: str,
```

and store it next to the ds4 assignment:

```python
        self._ds4_base_url = (ds4_base_url or "").strip()
        self._ollama_base_url = (ollama_base_url or "").strip() or "http://localhost:11434"
```

3d. `compose()` — after the existing ds4 `Input`, before the `OptionList`:

```python
        yield Static("ollama endpoint:", id="ollama_label")
        yield Input(value=self._ollama_base_url, placeholder="http://localhost:11434", id="ollama_url")
```

3e. Add the getter next to `_current_ds4_url`:

```python
    def _current_ollama_url(self) -> str:
        try:
            value = self.query_one("#ollama_url", Input).value.strip()
        except Exception:
            return self._ollama_base_url
        return value or "http://localhost:11434"
```

(An emptied field falls back to the default rather than disabling Ollama — unlike ds4, an empty Ollama URL has no useful meaning.)

3f. BOTH dismiss paths (`action_cancel` ~line 167 and `_save` ~line 301) — add after the `ds4_base_url=` line in each `SettingsResult(...)`:

```python
                ds4_base_url=self._current_ds4_url(),
                ollama_base_url=self._current_ollama_url(),
```

- [ ] **Step 4: Modify `archiver/app.py`**

4a. `action_settings` — pass the current value into the screen, after the `ds4_base_url=` line:

```python
                ds4_base_url=self.settings.ds4_base_url,
                ollama_base_url=self.settings.ollama_base_url,
```

4b. `_on_settings_done` — widen the change check and apply the field. Replace the `ds4_changed = ...` line with:

```python
        endpoints_changed = (
            result.ds4_base_url != self.settings.ds4_base_url
            or result.ollama_base_url != self.settings.ollama_base_url
        )
```

add to the `replace(...)` call, after `ds4_base_url=result.ds4_base_url,`:

```python
            ollama_base_url=result.ollama_base_url,
```

and rename the guard at the end of the method:

```python
        if endpoints_changed:
            self.run_worker(self._run_discovery())
```

- [ ] **Step 5: Extend the privacy guard**

Replace `scripts/check_no_private_host.sh` with the multi-endpoint version (still reading hostnames from the LOCAL config outside the repo — no hostname is ever written into this script):

```sh
#!/bin/sh
# Fails if a private hostname from the user's local config (the ds4 or ollama
# endpoint — that config lives outside the repo) appears in tracked files.
HOSTS=$(python3 -c '
import json, os
from urllib.parse import urlparse
path = os.path.expanduser("~/.config/amenity-stuff/config.json")
try:
    with open(path) as fh:
        cfg = json.load(fh)
except Exception:
    cfg = {}
for key in ("ds4_base_url", "ollama_base_url"):
    host = urlparse(cfg.get(key) or "").hostname or ""
    if host and host not in ("localhost", "127.0.0.1"):
        print(host)
')
if [ -z "$HOSTS" ]; then
    echo "OK (no private host configured)"
    exit 0
fi
STATUS=0
for HOST in $HOSTS; do
    if git grep -qiF "$HOST" -- . 2>/dev/null; then
        echo "LEAK: private hostname found in tracked files:"
        git grep -inF "$HOST" -- .
        STATUS=1
    fi
done
if [ "$STATUS" -eq 0 ]; then
    echo "OK"
fi
exit "$STATUS"
```

- [ ] **Step 6: Verify the guard still behaves, then run the full suite**

```bash
sh scripts/check_no_private_host.sh          # must print OK (or "OK (no private host configured)")
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v
~/.local/share/amenity-stuff/venv/bin/python -c "import archiver.app, archiver.settings_screen, archiver.ui_status"
```

Expected: guard OK; 49 tests pass; imports clean.

- [ ] **Step 7: Commit**

```bash
git add archiver/settings_screen.py archiver/app.py scripts/check_no_private_host.sh tests/test_settings_screen_result.py
git commit -m "feat: add ollama endpoint field to settings and guard its hostname"
```

---

### Task 4: Docs, version bump, manual PTY verification

**Runs LAST** — after Task 5 (Ollama truncation fix), so the release covers it.

**Files:**
- Modify: `README.md` (LLM Provider section; Settings list), `PROJECT_SPEC.md` (Configuration section), `CLAUDE.md` (Model Selection section)
- Modify: `VERSION` (`0.10.1` → `0.11.0`), `pyproject.toml` (version only), `CHANGELOG.md`

**Interfaces:**
- Consumes: everything above, complete.
- Produces: released docs + version.

- [ ] **Step 1: Update README.md**

Replace the detection list under `## LLM Provider`:

```markdown
On startup the app probes the configured endpoints:
- Ollama — `GET /api/tags` on the configured URL (default `http://localhost:11434`)
- the ds4 endpoint — `GET /v1/models`, if one is configured (see below)

Both servers may run on another machine on your local network: set their URLs in
Settings (`F2`). Nothing is sent outside the machines you point the app at.
```

In the `## Settings` bullet list, add:

```markdown
- ollama endpoint (default `http://localhost:11434`, may be a remote host)
```

- [ ] **Step 2: Update PROJECT_SPEC.md**

In the `## Configuration` section's "Configurable items include:" list, add:

```markdown
- LLM endpoints: Ollama URL (default `http://localhost:11434`) and the optional
  OpenAI-compatible ds4 URL — either may point at another machine on the local network
```

- [ ] **Step 3: Update CLAUDE.md**

In `### Model Selection ("auto")`, replace the first bullet:

```markdown
- `discovery.py` probes both providers over HTTP at startup (Ollama `GET /api/tags`,
  ds4 `GET /v1/models`); both endpoints are configurable and may be remote.
  `model_selection.py` merges their models into text/vision candidate lists
```

- [ ] **Step 4: Version bump + CHANGELOG**

- `VERSION`: `0.11.0`
- `pyproject.toml`: `version = "0.11.0"`
- `CHANGELOG.md`, new entry above `## [0.10.1]`:

```markdown
## [0.11.0] - 2026-08-01

- The Ollama endpoint is now configurable (Settings → ollama endpoint, default
  `http://localhost:11434`), so Ollama can run on another machine on the local
  network. Provider discovery now probes `GET /api/tags` over HTTP instead of
  shelling out to the `ollama` CLI, which is what makes remote hosts work.
- Fix silently degraded results with mid-size Ollama models: generation ceilings
  (`num_predict`) were tuned for 1B models and truncated larger ones mid-JSON,
  and a truncated reply (`done_reason: "length"`) was treated as success. The
  ceilings are now sized for 7-8B models, the normalization budget scales with
  the batch, and a truncated reply is reported as an error so the pipeline falls
  through to the next model candidate.
```

- [ ] **Step 5: Reinstall and run the automated suite**

```bash
~/.local/share/amenity-stuff/venv/bin/pip install -e .
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Manual PTY verification**

Per the project's rule, verify interactive behavior under a real PTY, not a pipe:

```bash
mkdir -p /tmp/amenity-sample && cp <a couple of .txt/.pdf samples> /tmp/amenity-sample/
script -qec "amenity-ai --source /tmp/amenity-sample --archive /tmp/amenity-archive" /dev/null
```

Checklist:
1. Default config (no `ollama_base_url` set): provider line shows `ollama` with the local model count — behavior identical to before.
2. `F2`: both endpoint fields render (ds4 above, ollama below); editing the ollama field and saving persists it to `~/.config/amenity-stuff/config.json` and triggers a fresh discovery.
3. Set the ollama field to an unreachable URL (e.g. `http://127.0.0.1:1`), save: provider line shows `ollama(down)`, and the app stays responsive with no traceback.
4. Restore the default, `S` to scan: facts extraction completes through the local Ollama.

- [ ] **Step 7: Final commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add README.md PROJECT_SPEC.md CLAUDE.md VERSION pyproject.toml CHANGELOG.md
git commit -m "docs: document configurable ollama endpoint; release 0.11.0"
```

---

### Task 5: Surface Ollama truncation and right-size the token budgets

**Runs BEFORE Task 4** (Task 4 releases the version and its CHANGELOG entry covers this fix too).

**Context — why this task exists:** e2e verification against a real remote Ollama (an 8B model) showed facts extraction returning empty content on a perfectly readable document. Root cause, with evidence from the live server: `num_predict: 400` truncates the facts JSON mid-string (`done_reason: "length"`, `eval_count: 400`, output unparseable), and `OllamaBackend.generate` ignores `done_reason`, so the truncated text flows into the JSON-repair path, which "succeeds" with empty fields instead of failing over to the next candidate. Same defect class as the ds4 `finish_reason: "length"` fix already shipped in 0.10.0 — the Ollama side never got it. The caps were tuned for tiny models (`gemma3:1b`); any mid-size model degrades silently today. `num_predict` is a ceiling, not a target, so raising it costs nothing for models that finish early.

**Files:**
- Modify: `archiver/ollama_client.py` (`OllamaBackend.generate` response handling)
- Modify: `archiver/analyzer.py` (`_JSON_REPAIR_OPTIONS` line 211, `_FACTS_GENERATE_OPTIONS` line 212, `_CLASSIFY_GENERATE_OPTIONS` line 213)
- Modify: `archiver/normalizer.py` (`_NORMALIZE_GENERATE_OPTIONS` line 31 → batch-scaled helper; its use at line 432)
- Test: `tests/test_ollama_truncation.py` (create)

**Interfaces:**
- Consumes: `LLMResponse` from `archiver/llm_backend.py`.
- Produces: `normalizer._normalize_options(batch_size: int) -> dict` (module-level helper replacing the `_NORMALIZE_GENERATE_OPTIONS` constant).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ollama_truncation.py`:

```python
from __future__ import annotations

from archiver import ollama_client
from archiver.normalizer import _normalize_options
from archiver.ollama_client import OllamaBackend


def _response(text: str = '{"ok": true}', done_reason: str = "stop") -> dict:
    return {"model": "qwen3:8b", "response": text, "done": True, "done_reason": done_reason}


def test_generate_flags_truncated_output_as_error(monkeypatch):
    monkeypatch.setattr(
        ollama_client, "_post_json",
        lambda url, payload, *, timeout_s: _response('{"partial": "cut off mid-str', done_reason="length"),
    )
    resp = OllamaBackend("http://localhost:11434").generate(prompt="q", model="qwen3:8b")
    assert not resp.success
    assert "truncated" in (resp.error or "")


def test_generate_accepts_normal_completion(monkeypatch):
    monkeypatch.setattr(
        ollama_client, "_post_json",
        lambda url, payload, *, timeout_s: _response(),
    )
    resp = OllamaBackend("http://localhost:11434").generate(prompt="q", model="qwen3:8b")
    assert resp.success
    assert resp.text == '{"ok": true}'


def test_generate_without_done_reason_still_succeeds(monkeypatch):
    # Older Ollama versions omit done_reason entirely.
    monkeypatch.setattr(
        ollama_client, "_post_json",
        lambda url, payload, *, timeout_s: {"model": "gemma3:1b", "response": "hi", "done": True},
    )
    resp = OllamaBackend("http://localhost:11434").generate(prompt="q", model="gemma3:1b")
    assert resp.success


def test_normalize_options_scale_with_batch_size():
    single = _normalize_options(1)
    batch = _normalize_options(12)
    assert single["temperature"] == 0
    assert single["num_predict"] >= 800
    assert batch["num_predict"] >= 12 * 250       # one row per item needs its own budget
    assert batch["num_predict"] > single["num_predict"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_ollama_truncation.py -v`
Expected: FAIL — `cannot import name '_normalize_options'`; and the truncation test fails because `generate` currently returns success for `done_reason: "length"`

- [ ] **Step 3: Surface truncation in `archiver/ollama_client.py`**

In `OllamaBackend.generate`, replace the success-path return with a truncation check first. The current block is:

```python
            data = _post_json(url, payload, timeout_s=timeout_s)
            error = data.get("error") if isinstance(data.get("error"), str) else None
            return LLMResponse(
                text=str(data.get("response", "")),
                model=data.get("model"),
                done=data.get("done", True),
                error=error,
            )
```

becomes:

```python
            data = _post_json(url, payload, timeout_s=timeout_s)
            error = data.get("error") if isinstance(data.get("error"), str) else None
            if error is None and data.get("done_reason") == "length":
                # Hit the num_predict ceiling: the payload is cut mid-token and any
                # JSON in it is unparseable. Report it so the caller falls through to
                # the next candidate instead of "repairing" truncated garbage.
                error = "ollama: output truncated by num_predict"
            return LLMResponse(
                text="" if error else str(data.get("response", "")),
                model=data.get("model"),
                done=data.get("done", True) and error is None,
                error=error,
            )
```

- [ ] **Step 4: Right-size the budgets in `archiver/analyzer.py`**

Replace the three constants (lines 211-213) with:

```python
# num_predict is a ceiling, not a target: models that finish earlier cost nothing
# extra. These are sized so a mid-size model (7-8B) can emit a complete JSON object;
# the previous values were tuned for 1B models and truncated everything larger.
_JSON_REPAIR_OPTIONS = {"temperature": 0, "num_predict": 1500}
_FACTS_GENERATE_OPTIONS = {"temperature": 0, "num_predict": 2000}
_CLASSIFY_GENERATE_OPTIONS = {"temperature": 0, "num_predict": 1200}
```

(The repair call must re-emit a whole facts object, so it needs a budget in the same league as facts itself — 220 could not have repaired anything but the shortest output.)

- [ ] **Step 5: Scale the normalize budget with batch size in `archiver/normalizer.py`**

Replace the constant at line 31:

```python
_NORMALIZE_GENERATE_OPTIONS = {"temperature": 0, "num_predict": 220}
```

with a helper:

```python
def _normalize_options(batch_size: int) -> dict:
    """Token ceiling for one normalization call.

    Each item costs a JSON row (category, year, proposed name, summary), so the
    budget has to grow with the batch: a fixed 220 could not fit even two rows,
    which silently truncated every batch and forced the per-item fallback.
    """
    return {"temperature": 0, "num_predict": max(800, 300 * max(1, batch_size))}
```

and at the `generate(...)` call (line ~432):

```python
            options=_normalize_options(len(batch)),
```

- [ ] **Step 6: Run the full suite**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: all PASS (previous total + 4 new). If a pre-existing test asserted the old constants, update it to the new values and say so in your report.

- [ ] **Step 7: Commit**

```bash
sh scripts/check_no_private_host.sh   # must print OK
git add archiver/ollama_client.py archiver/analyzer.py archiver/normalizer.py tests/test_ollama_truncation.py
git commit -m "fix: surface ollama truncation and right-size token budgets"
```

---

## Self-Review Notes

- Spec coverage: config+threading (Task 2), HTTP discovery incl. removal of CLI/`_run`/`shutil` (Task 1), settings screen + re-discovery + extended guard (Task 3), `ollama(down)` label (Task 1 Step 6), docs+0.11.0+manual checks (Task 4). Adaptation of the existing ds4 discovery tests to the new `_discover_ollama` signature is Task 1 Step 5, as the spec's testing section requires.
- Deliberate asymmetry with ds4, called out where it appears: `ollama_base_url` defaults to `http://localhost:11434` and an emptied field falls back to that default, whereas `ds4_base_url` defaults to `""` and empty means disabled.
- `ProviderInfo.command` removal (Task 1 Step 4) goes marginally beyond the spec's wording but is the direct consequence of dropping the CLI path; grep confirms no reader outside `discovery.py`.
