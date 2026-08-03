# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.15.1] - 2026-08-03

- The settings screen stops wasting the taxonomy's space. The provider
  summary loses its border and padding, each provider becomes a single row —
  name, URL, parallel slots — instead of five lines of label, blanks and
  bordered input, and the blank line above the taxonomy label is gone. On a
  40-row terminal the taxonomy editor grows from ~4 rows to 20.
- The language tag in the taxonomy label ("[EN]", "[IT]") is visible again:
  Rich was silently eating it as a markup style tag.

## [0.15.0] - 2026-08-03

- Three more file types, each riding plumbing that already existed:
  - `.webp` joins the image pipeline (vision + OCR). The vision path already
    sniffed its MIME type from the magic bytes; the extension only had to be
    let in. One real scan counted 17 of these sitting unscanned.
  - `.ods` reuses the odt extractor: every ODF document keeps its text in
    the same `content.xml` container, so spreadsheet cells come out with the
    machinery that was already there.
  - `.ics` gets a small stdlib extractor: calendar name and events with
    summary, date, location and description. Dates are rewritten with dashes
    ("20240712" → "2024-07-12") because the analyzer's year hint refuses a
    year followed by more digits — the raw form would feed it nothing.

## [0.14.0] - 2026-08-03

- Two new file types, both surfaced by a real scan that found 6 of one and 11
  of the other with no extractor:
  - `.pptx` — slide text plus title, author and creation date from the
    document properties; that date is exactly the year hint the analyzer
    feeds on. Standard library only (a pptx is a zip of XML, like kmz).
    Legacy binary `.ppt` is not covered.
  - `.eml` — headers, the preferred body part (plain text, or HTML stripped
    of tags) and attachment names, which often name the actual document
    better than the email text does. The `Date` header supplies the year.
- Both are in the default `include_extensions`, so previously skipped files
  are picked up by `k` + `S` after upgrading.

## [0.13.2] - 2026-08-03

- csv, html, yaml and gpx files are scanned again. Their extractors have
  existed all along, but the analyzer kept its own twin list of supported
  kinds and the two had drifted apart, so these files were reported
  "Unsupported file type" — 40 of the 68 skips in one real 950-file scan.
  The analyzer now derives its dispatch from the extractor registry, and a
  test pins the whole chain so the lists cannot drift again.
- `k` requeues skipped and error rows as pending, touching nothing else.
  Skips are often systematic — a file type the analyzer could not reach, a
  provider that was down — and the only remedies were `r` once per row or
  `R`, which throws away every good result along with the bad ones. Only
  rows an extractor can actually handle are requeued: the zips and webp the
  scanner lists as unsupported stay where they are, since another pass
  cannot change their outcome.

## [0.13.1] - 2026-08-03

- Classification sends its chunks concurrently too, through the same
  per-provider limit as the scan. Thirty-six files in chunks of twelve went
  from 154s to 59s, with identical categories, years and names.
- Chunk size is deliberately unchanged: a longer prompt is a longer request
  with more output and worse answers. Three requests of twelve at once beats
  one request of thirty-six.
- Each chunk is now independent. A chunk that failed used to end the whole
  pass and discard every chunk after it; the healthy ones now still produce
  their results, and the error is reported alongside them. A cancelled run
  still reports itself as cancelled ahead of any other error.
- Fixes a latent bug the same change exposed: the single-item recovery path
  tested a dictionary already filled by earlier chunks, so it could never fire
  after the first one.

## [0.13.0] - 2026-08-03

- The scan phase now works on several files at once instead of one at a time.
  On a real vLLM the measured knee is four concurrent requests: below it the
  server is idle, above it throughput stops improving and per-file latency
  grows. A twelve-file run went from 314s to 90s.
- The limit belongs to the provider, not to the app: vLLM defaults to 4,
  Ollama and ds4 to 1, and each is editable next to its endpoint in Settings.
  ds4 answers one caller at a time, so nothing may flood it; for Ollama we
  cannot see `OLLAMA_NUM_PARALLEL`, so the conservative default stands until
  you raise it.
- The status line reports the run rather than the table: queued, in flight,
  done, plus throughput and an estimated time left. Stopping now says how many
  requests it is waiting for, since a request already sent cannot be recalled.
  Work already finished when you stop is kept, not discarded.
- The banner judges the setup by role instead of by Ollama alone. With Ollama
  deliberately stopped and vLLM covering both roles it showed a permanent red
  error; a provider that is down while the roles are covered is now a warning.
- A file that crashes during extraction is marked as an error and the run
  carries on. It used to escape the loop and leave the interface stuck on
  "running" for good.
- Cache writes are batched instead of one per file, so a fast run cannot stall
  the interface by re-serialising the whole cache between files.

## [0.12.3] - 2026-08-02

- Everything the app says is in English again. The provider redesign had
  introduced Italian in the doctor screen, the check messages, the router
  errors and the model catalogue, while the rest of the interface was
  English. Comments and docstrings in the modules added by that work follow.
  The Italian stopword lists stay: those are data, not prose.

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
