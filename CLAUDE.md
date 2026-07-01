# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**amenity-ai** is a Python TUI (Textual framework) that organizes files using local LLM analysis via Ollama. It follows a 2-phase workflow:
1. **Scan**: Extract content from files (PDF, images, Office docs, text, CSV, HTML, GPX, YAML) + optional OCR, then call LLM to produce structured facts
2. **Classify**: Use facts + user-defined taxonomy to propose category, reference year, and filename

Files can then be moved to an archive structured as `{category}/{year}`.

See `EXTRACTORS.md` for supported formats and how to add new ones.

## Commands

```bash
# One-line install (for users)
curl -sSL https://raw.githubusercontent.com/elmisi/amenity-ai/main/install.sh | sh

# Uninstall
curl -sSL https://raw.githubusercontent.com/elmisi/amenity-ai/main/uninstall.sh | sh

# Development: refresh local install after version bump
~/.local/share/amenity-stuff/venv/bin/pip install -e .

# Quick test without installing
python3 -m archiver

# Run the TUI
amenity-ai --source /path/to/folder --archive /path/to/archive

# Run from source directly
python3 -m archiver

# Generate performance report from cache
amenity-ai report --source /path/to/folder

# Bump version before committing code changes
python3 scripts/bump_version.py
```

**No test suite currently exists.** There is no Makefile, pytest, or linting configuration.

## Architecture

### Module Organization
- **archiver/app.py**: Main Textual TUI application
- **archiver/analyzer.py**: LLM facts/classification logic (analysis pipeline)
- **archiver/extractors/**: File format handlers (PDF, Office, images, text formats)
  - `registry.py` dispatches based on file kind
  - `filetypes.py` maps extensions to kinds
  - See `EXTRACTORS.md` for adding new formats
- **archiver/taxonomies/**: Default taxonomy files (en.txt, it.txt)
  - Users can override in `~/.config/amenity-stuff/taxonomies/`
- **archiver/prompts.py**: All LLM prompt templates (facts, classification)
- **archiver/llm_backend.py**: LLM interface abstraction (`LLMBackend` protocol)
- **archiver/ollama_client.py**: Ollama HTTP API wrapper
- **archiver/normalizer.py**: Normalization of LLM responses
- **archiver/utils_parsing.py**: Text/date/amount parsing utilities
- **archiver/utils_filename.py**: Filename manipulation helpers
- **archiver/ui_*.py**: UI rendering and formatting helpers (separated from logic)
- **archiver/*_screen.py**: Textual screen widgets (settings, help, confirm, etc.)
- **archiver/scanner.py**: File discovery and `ScanItem` dataclass
- **archiver/cache.py**: Persistent result caching keyed by `(path, size, mtime)`
- **archiver/config.py**: User configuration (~/.config/amenity-stuff/config.json)

### Key Patterns
- **UI/logic separation**: Analysis, extraction, and LLM code live outside UI layer
- **Async workers**: Long-running operations (scan, OCR, LLM) run in workers; UI updates via `call_from_thread`
- **Frozen dataclasses**: Used for configs, results, and state to prevent accidental mutations
- **Cache validity**: Checked by `(path, size, mtime)` before reusing results

### Data Flow
```
File → Extractor → Text/Content → LLM (facts) → Cache
                                      ↓
Cached facts + Taxonomy → LLM (classify) → Category/Year/Name proposal
                                      ↓
User approval → Move to archive/{category}/{year}/
```

## Developer Guidelines

See **AGENTS.md** for detailed conventions. Key points:

- **Never block the Textual event loop** with I/O, OCR, or LLM calls
- Keep refactors behavior-preserving; move code in small steps
- Bump **patch version** only for Python code changes in `archiver/`: `python3 scripts/bump_version.py`
- Do NOT bump version for: docs, shell scripts, config files
- Commit messages: `type: description` (e.g., `fix: skip facts when summary_long missing`)
- Use `pathlib.Path` for filesystem paths; add type hints on public functions

### SOLID Principles in Practice
- **New extractors**: Add module in `extractors/`, register in `registry.py` (no changes to existing code)
- **New LLM backends**: Implement `LLMBackend` protocol from `llm_backend.py`
- **Prompts**: Externalized in `prompts.py` for easy customization
- High-level modules depend on abstractions; configuration is injected via dataclasses
