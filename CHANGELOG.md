# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.12.2] - 2026-08-02

- Remove an analysis path that no code reached: `analyze_item` had no callers
  anywhere in the repository, and with it went `_try_text_models`,
  `_classify_from_text`, the `AnalysisResult` dataclass and the classify prompt
  they were the only users of. Classification runs through `normalizer.py`,
  which has its own prompt. 327 lines of unreachable code, no behaviour change.

## [0.12.1] - 2026-08-02

- Stop gluing ordinary short words together in proposed filenames. The repair
  that fixes a word broken by a stray separator ("Mi_iti" → "Miiti") fired on
  any word of three letters or less, so `per il mese` became `peril mese`,
  `for the office` became `forthe office` and `via roma` became `viaroma`. It
  now looks for what actually identifies a broken word — a short capitalised
  fragment followed by a lowercase tail — instead of length alone, and reuses
  the existing `STOPWORDS` list rather than a shorter second copy of it.

## [0.12.0] - 2026-08-01

### Added
- vLLM is now a first-class provider alongside Ollama and ds4. Configuration is
  three endpoints and nothing else: the app discovers the available models by
  itself, in parallel, one request per provider.
- `amenity-ai doctor`, and `d` in the TUI: checks every provider and reports
  whether a semantic model and a vision model are actually usable. When a model
  is missing it offers a curated shortlist with sizes and can install it —
  bearing in mind the download happens on the machine hosting Ollama, not on
  yours. Cancel with `x`. Exit code is non-zero when a check fails.
- Vision now works on OpenAI-compatible servers. Image captioning used to talk
  to Ollama directly, so a multimodal model served by vLLM was unusable.
- Reasoning models are asked not to reason when there is nothing to reason
  about. Each provider gets its own lever, because they silently ignore each
  other's: vLLM takes `chat_template_kwargs.enable_thinking`, ds4 takes
  `reasoning_effort`, Ollama has its native `think`. Captioning switches it off
  too, which it previously did not. Measured on one photo through the whole
  pipeline: 168s to 17s, with an equivalent summary.

### Fixed
- Pictures that are not documents are no longer discarded. The facts prompt asked
  a paper-document question, so a photo or a symbol came back with a perfectly
  good description *and* a `skip_reason` saying there were no invoice-like
  fields — and the whole answer was thrown away. Captions now yield a subject,
  tags and a summary, so the file gets a real name instead of none.

### Changed
- Model selection is driven by real metadata instead of three hardcoded
  preference lists: provider priority (`vllm > ollama > ds4`) first, then a size
  band chosen per role, then a hand-maintained list used only to break ties.
- Every model id now carries its provider prefix (`ollama:`, `vllm:`, `ds4:`).
- Capabilities come from the provider when it declares them, from a live probe
  when the doctor runs, and only otherwise from the model name. The probe result
  is remembered, because guessing from the name has real false negatives: a
  multimodal model served over the OpenAI API need not say so in its name.

### Removed
- The "vision fallback" setting: the ranking already produces an ordered list of
  vision candidates, so a separate second choice had nothing left to add.

### Migration
Existing configurations are migrated automatically on first load: the two flat
endpoint fields become the new mapping, and bare model ids gain the `ollama:`
prefix. No manual action is required.

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

## [0.10.1] - 2026-08-01

- Update dependency ranges to current major versions: textual 8.x (from 0.6x),
  pypdf 6.x, PyMuPDF 1.28, Pillow 12.x. No code changes required; verified
  with the full test suite and an interactive TUI pass (scan, settings, quit).

## [0.10.0] - 2026-08-01

- Add support for a second local LLM provider: any OpenAI-compatible
  chat-completions server ("ds4"), configured via Settings → ds4 endpoint.
  Models appear with the `ds4:` prefix, are preferred for facts/classify in
  `auto` mode, and fall back to Ollama when the server is unavailable.
  Text-only: vision stays on Ollama.

## [0.9.13] - 2026-07-01

### Changed
- Renamed the project from `amenity-stuff` to `amenity-ai`. This affects the GitHub
  repository, the Python package name, and the CLI command (`amenity-stuff` → `amenity-ai`).
- Updated documentation, install/uninstall scripts, and taxonomy file headers to the new name.

### Notes
- Runtime directories are intentionally **unchanged** for backward compatibility, so existing
  settings and caches keep working:
  - user config: `~/.config/amenity-stuff/`
  - per-source / per-archive caches: `.amenity-stuff/`
  - install location: `~/.local/share/amenity-stuff/`
- After upgrading, reinstall to pick up the new command name:
  `~/.local/share/amenity-stuff/venv/bin/pip install -e .`
