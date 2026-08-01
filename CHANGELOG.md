# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
