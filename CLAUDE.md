# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**amenity-ai** is a Python TUI (Textual framework) that organizes files using local LLM analysis via Ollama. It follows a 2-phase workflow:
1. **Scan**: Extract content from files (PDF, images incl. webp, Office docs, PowerPoint, email, calendar, text, CSV, HTML, GPX, YAML) + optional Tesseract OCR, then call LLM to produce structured facts
2. **Classify**: Use cached facts + user-defined taxonomy to propose category, reference year, and filename

Files can then be moved to an archive structured as `{category}/{year}` (or `{category}/{undated}`).

**Naming quirk**: the Python package is `archiver/`, the CLI command is `amenity-ai`, and config/cache directories use `amenity-stuff` (`~/.config/amenity-stuff/`, `<source>/.amenity-stuff/`, `~/.local/share/amenity-stuff/venv`).

Related docs: `AGENTS.md` (detailed conventions), `EXTRACTORS.md` (adding file types), `PROJECT_SPEC.md` (spec + roadmap).

## Commands

```bash
# Quick test without installing
python3 -m archiver

# Run the TUI
amenity-ai --source /path/to/folder --archive /path/to/archive

# Generate performance report from cache
amenity-ai report --source /path/to/folder

# Check providers and models (exit code != 0 when a check fails)
amenity-ai doctor

# Development: refresh local install after changes (updates the system-wide command)
~/.local/share/amenity-stuff/venv/bin/pip install -e .

# Bump patch version before committing Python code changes (updates VERSION + pyproject.toml)
python3 scripts/bump_version.py

# One-line install / uninstall (for users)
curl -sSL https://raw.githubusercontent.com/elmisi/amenity-ai/main/install.sh | sh
curl -sSL https://raw.githubusercontent.com/elmisi/amenity-ai/main/uninstall.sh | sh
```

A pytest suite exists in `tests/` (unit tests for the LLM routing/config layers). Run it with
`~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v` (pytest is dev-only,
deliberately not in `pyproject.toml`). There is still no Makefile or linting configuration.

## Architecture

### Data Flow
```
File → Extractor → Text/Content → LLM (facts) → Cache
                                      ↓
Cached facts + Taxonomy → LLM (classify) → Category/Year/Name proposal
                                      ↓
User approval → Move to archive/{category}/{year}/
```

### Per-file State Machine
`ScanItem` (frozen dataclass in `scanner.py`) is the single source of truth per file; the table is a view of it. `status` drives everything: `pending → scanned → classified → moved`, plus `skipped`/`error`. Transient statuses (`scanning`, `classifying`, `moving`) are never restored from cache — `cache_overlay.py` maps them (and legacy status names) back to stable ones when overlaying cached results onto a fresh scan.

### Two Configuration Dataclasses
- `config.py` → `AppConfig`: persisted at `~/.config/amenity-stuff/config.json` via `load_config()`/`save_config()`; `load_config()` migrates legacy keys (old `text_model`, `taxonomy_lines`, the flat `ollama_base_url`/`ds4_base_url` pair, and bare model ids via `migrate_model_id()`)
- `settings.py` → `Settings`: frozen runtime settings, assembled in `__main__.py` from CLI args + `AppConfig`
- Both carry `providers: dict[str, str]` — one URL per registry entry, normalized in `__post_init__` so unknown keys are dropped and missing ones fall back to the registry default. An empty URL means "provider disabled"; endpoints are user-local and must never be hardcoded in the repo

Adding a user-facing setting means touching both dataclasses, `__main__.py` wiring, and `settings_screen.py`.

### Provider Registry
`providers.py` is the single source of truth. A `ProviderSpec` carries `name`, `kind`
(`ollama` | `openai_compat`), `prefix`, `default_url`, `sends_reasoning_effort`,
`supports_install`. **Declaration order is the priority**: `vllm > ollama > ds4` — vLLM
batches concurrent requests, Ollama is always available, ds4 is mutually exclusive so a
long scan would monopolize it.

Every model id carries its provider prefix (`ollama:`, `vllm:`, `ds4:`) through candidates,
config, cache `model_used`, and the UI. `split_model_id()` matches against known prefixes
rather than splitting on the first `:`, so `ollama:qwen3:8b` resolves correctly and a bare
`qwen3:8b` (written by a config older than 0.12.0) means Ollama.

Adding a provider means one entry in `PROVIDERS`; discovery, ranking, routing, settings and
the doctor all iterate the registry.

### Model Selection ("auto")
- `discovery.py` queries every configured provider in parallel, one request each (Ollama
  `GET /api/tags`, OpenAI-compatible `GET /v1/models`), and returns `ModelInfo` with
  capabilities, parameter size and context length. Ollama ≥ 0.31 declares capabilities in
  `/api/tags`, so no per-model request is needed
- Capabilities come from three tiers of decreasing trust: `declared` (from the provider) >
  `probed` (a live 1×1 PNG request, only inside the doctor, persisted in `probe_cache.py`) >
  `heuristic` (name patterns in `capabilities.py`). The heuristic has real false negatives:
  a multimodal model served over the OpenAI API need not say so in its name
- `model_selection.rank_models(models, role)` orders candidates for `facts`, `classify` and
  `vision` by four criteria: provider priority → size bucket (`facts`/`vision` ascending,
  `classify` closest to the 5–9B band) → position in `CURATED_BIAS` → id. `CURATED_BIAS`
  only breaks ties inside a bucket and is maintained by hand
- `task_builders.build_analysis_config()` applies the pinned model, if any, on top of the
  ranking without dropping the others; analysis tries candidates in order until one succeeds
- LLM calls pin `temperature=0`, JSON response format, `keep_alive="5m"`, and capped
  `num_predict` (constants in `analyzer.py`); content is excerpted head+tail to ~10k chars
- `llm_router.generate(..., provider_urls=...)` resolves the prefix to a backend:
  `ollama_client.OllamaBackend` or `openai_client.OpenAICompatBackend`. An unconfigured
  provider fails explicitly rather than falling back. Vision goes through the router too,
  so a multimodal model on vLLM is usable

### Doctor
`doctor.py` is pure: it takes a `DiscoveryResult` and returns a `DoctorReport` of `Check`s —
one per provider plus one per semantic role. The probe is injected as a callable, so tests
never touch the network and the UI decides whether to pay its cost. A `Remedy` is data, not
a string: `kind="pull"` only when a provider with `supports_install=True` is reachable,
otherwise it degrades to `kind="hint"`. Two surfaces, one logic: `amenity-ai doctor` and
`[d]` in the TUI. The doctor never opens by itself.

### Caches
- Source cache: `<source>/.amenity-stuff/cache.json`, keyed by `(path, size, mtime)`; checked before reusing results
- On move: entry is written to `<archive>/.amenity-stuff/cache.json`, source entry kept with status `moved` (+ `moved_to`), and an append-only log goes to `<archive>/.amenity-stuff/moves.jsonl`
- Invalidation is explicit and user-driven (`r` reset row, `R` reset all, `u` unclassify keeps facts but clears classification)

### Task Orchestration
- One long-running task (scan/classify/move) at a time, tracked by `TaskState` (`task_state.py`). Within the facts phase, files run through a `ThreadPoolExecutor`; the real regulator is `concurrency.py`'s `ConcurrencyLimiter` — one semaphore per provider, acquired inside `llm_router.generate` because a candidate fallback can send two files of the same run to different providers. Limits come from `ProviderSpec.max_concurrency` and are overridable per provider in Settings
- Cancellation is cooperative: `x` sets `cancel_requested`; workers poll it (`should_cancel` callbacks) and stop between items
- **Never block the Textual event loop** with I/O, OCR, or LLM calls — run them in workers; UI updates via `call_from_thread`

### Module Organization
- **archiver/app.py**: Main Textual TUI application (largest file; keybindings, workers, table wiring)
- **archiver/analyzer.py**: LLM facts/classification pipeline (`AnalysisConfig`, prompt calls, JSON repair)
- **archiver/extractors/**: File format handlers — `../filetypes.py` maps extension → kind, `registry.py` dispatches kind → extractor
- **archiver/prompts.py**: All LLM prompt templates (facts, classification)
- **archiver/providers.py**: Provider registry — names, prefixes, priority, per-provider quirks
- **archiver/concurrency.py**: `ConcurrencyLimiter` (one semaphore per provider) and the pool sizing for a run
- **archiver/capabilities.py** / **probe_cache.py**: Capability detection and size parsing / persistence of conclusive probe results
- **archiver/discovery.py** / **model_selection.py**: Parallel model discovery / role-based ranking
- **archiver/llm_backend.py** / **ollama_client.py** / **openai_client.py**: `LLMBackend` protocol abstraction / Ollama HTTP wrapper / shared OpenAI-compatible backend (vLLM and ds4)
- **archiver/doctor.py** + **model_catalog.py** + **ollama_admin.py**: Diagnosis logic / curated installable models / model pull and vision probe
- **archiver/normalizer.py**: Normalization of LLM responses
- **archiver/taxonomy.py** + **archiver/taxonomies/**: Taxonomy parsing + bundled defaults (en/it); user overrides in `~/.config/amenity-stuff/taxonomies/{lang}.txt`
- **archiver/archive_apply.py**: Move-to-archive logic
- **archiver/utils_parsing.py** / **utils_filename.py**: Text/date/amount parsing; filename manipulation
- **archiver/ui_*.py**: Rendering/formatting helpers (kept separate from logic)
- **archiver/*_screen.py**: Textual screens (settings, help, confirm, setup, archive picker)

### Adding a New File Type (Open/Closed)
Four touch points, no changes to existing extractors — see `EXTRACTORS.md` for the full walkthrough:
1. `archiver/filetypes.py`: map extension → kind
2. `archiver/extractors/textish_<kind>.py`: new module returning `Optional[str]` (or `_with_meta` variant); lazy-import heavy deps, return `None` on failure, respect `max_chars`
3. `archiver/extractors/registry.py`: route the kind
4. `archiver/settings.py`: add to `include_extensions`

## Developer Guidelines

See **AGENTS.md** for detailed conventions. Key points:

- Keep refactors behavior-preserving (don't alter prompts, heuristics, defaults, or UX flows); move code in small steps
- UI/logic separation: analysis, extraction, and LLM code live outside the UI layer; prefer "data in / data out" functions
- Frozen dataclasses for configs, results, and state
- Use `pathlib.Path` for filesystem paths; type hints on public functions
- Bump **patch version** only for Python code changes in `archiver/`: `python3 scripts/bump_version.py` (VERSION and `pyproject.toml` must stay in sync). Do NOT bump for docs, shell scripts, or config files
- Commit messages: `type: description` (e.g., `fix: skip facts when summary_long missing`, `tui: refactor settings screen`)
- New LLM backends: implement the `LLMBackend` protocol from `llm_backend.py`; high-level modules depend on abstractions, configuration is injected via dataclasses
