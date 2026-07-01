# Project Specification (amenity-ai)

`amenity-ai` is a local-first Terminal UI to analyze and organize files using a local LLM (via Ollama).

The long-term goal is to help users turn a “messy folder” into a clean archive structure with meaningful names, using an interactive review/approval workflow.

## Scope (current MVP)

### Supported file types
- Documents: `.pdf`, `.doc/.docx/.odt`, `.xls/.xlsx`
- Images: `.jpg/.jpeg/.png`
- Text-ish: `.txt/.md/.json/.rtf/.svg/.kmz`
- Unsupported formats are still listed in the table and will be marked `skipped` with reason `unsupported file type`.

### 2-phase workflow
1. **Scan (facts extraction)**  
   For each file, extract the highest-signal content available:
   - PDF text extraction when possible
   - OCR via Tesseract for scanned PDFs / images (configurable)
   - Local LLM call (Ollama) to produce structured “facts” and a rich summary
   - No category/year/rename decision is required at this stage

2. **Classify**  
   Using only scanned results, propose:
   - taxonomy-driven **category**
   - best-effort **reference year**
   - a coherent **proposed filename** (without redundantly repeating category/year when not necessary)

## Non-goals (for now)
- No remote providers are supported (local Ollama only).
- No automatic “approve plan and apply” flow yet (moving is manual).

## UX Requirements (TUI)

### Table + details panel
- The main list shows one row per file with a compact status indicator.
- A details panel shows the full information for the selected row (non-truncated).
- Pressing `Enter` opens the selected file with the system default application (best-effort, silent failure).

### Operations and cancellation
- Long-running work runs in background workers; the UI stays interactive.
- A single scan/classify operation runs at a time.
- `x` stops the current task (scan, classify, move) as soon as possible (cooperative cancellation).

### Reset vs unclassify
- **Reset** (`r`) returns a file to `pending` and clears cached results (scan + classification).
- **Unclassify** (`u`) keeps scan results but clears classification fields, returning the item to `scanned`.

## Configuration

Settings are managed via the TUI (`F2`) and persisted in:
`~/.config/amenity-stuff/config.json`

Configurable items include:
- archive folder
- output language (`auto`, `it`, `en`)
- taxonomy (categories defined by the user)
- text/vision model selection (Ollama models)
- filename separator (space vs underscore)
- OCR mode

## Cache (MVP)

Per-source cache lives in:
`<source>/.amenity-stuff/cache.json`

Cache key: `(relative path, size, mtime)`  
Transient statuses are never reused from cache (e.g. `scanning`, `classifying`).

## Safety & Privacy
- Local-first by design: content is processed locally.
- If OCR is enabled, text is extracted from scanned content before LLM analysis.
- If you configure different models/providers in the future, review their data handling.

## Roadmap / Upcoming milestones
1. **Approval + apply phase**
   - per-file approval
   - dry-run first
   - move/rename into `{category}/{year}` under the archive root (and support for undated)
   - collision handling and filename sanitization
2. **Better extraction**
   - improved PDF text extraction and OCR heuristics
   - better resilience on low-quality scans
3. **Quality improvements**
   - stronger naming consistency across similar documents
   - better year inference (content + filename + metadata)
