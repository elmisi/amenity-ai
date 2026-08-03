# amenity-ai

Terminal UI to organize files using a local LLM (via Ollama) with a 2-phase workflow:
1) extract high-signal facts (no classification yet),
2) classify + propose coherent file names (taxonomy-driven).

You can then move files into an archive structured as `{category}/{year}` (or `{category}/{undated}`).

See `PROJECT_SPEC.md` for a more detailed (and up-to-date) project specification.

## Install

One-line install (recommended):
```bash
curl -sSL https://raw.githubusercontent.com/elmisi/amenity-ai/main/install.sh | sh
```

Uninstall:
```bash
curl -sSL https://raw.githubusercontent.com/elmisi/amenity-ai/main/uninstall.sh | sh
```

### Alternative: manual install

From source (development):
```bash
python3 -m venv .venv
./.venv/bin/pip install .
amenity-ai
```

System-wide via pipx:
```bash
pipx install git+https://github.com/elmisi/amenity-ai.git
amenity-ai
```

## Run

Pass source/archive on the CLI (defaults: `--source .` and `--archive ./ARCHIVE`):
```bash
amenity-ai --source /path/to/folder --archive /path/to/archive
```

## Performance report

After running Scan/Classify on a folder, you can print a short timing summary from the cache:

```bash
amenity-ai report --source /path/to/folder
```

## Doctor

Checks whether the configured providers are reachable and whether you actually have
a semantic model and a vision model to work with:

```bash
amenity-ai doctor
```

If something is missing it offers a short list of models with their sizes and can install
them for you. The download happens on the machine hosting Ollama, not on the one running
amenity-ai — point the endpoint at another host and the gigabytes land there. Press `x` to
cancel a download, `r` to re-check, `Esc` to close. The exit code is non-zero when a check
fails, so the command is usable in a script.

The same screen is available inside the TUI with `d`.

## Settings

You can change:
- output language (`auto`, `it`, `en`)
- taxonomy (allowed categories)
- models (facts / classify / vision), archive folder, filename separator, OCR mode
- one endpoint per provider: `ollama` (default `http://localhost:11434`), `vllm`, `ds4`.
  Any of them may point at another machine; leave one empty to disable it

You only configure the endpoints — amenity-ai discovers the available models by itself.

Press `F2` in the TUI to open Settings. Configuration is stored in `~/.config/amenity-stuff/config.json`.

### Taxonomies

Taxonomies are language-specific: when you change the output language, the taxonomy editor shows the categories for that language. Default taxonomies are provided for English and Italian.

Taxonomy format (one per line):
`name | description | examples` (examples are optional, separated by `;`).

**External taxonomy files** are loaded from (in order):
1. `~/.config/amenity-stuff/taxonomies/{lang}.txt` (user override)
2. `archiver/taxonomies/{lang}.txt` (bundled defaults)

To customize, copy the bundled file and edit:
```bash
mkdir -p ~/.config/amenity-stuff/taxonomies
cp archiver/taxonomies/it.txt ~/.config/amenity-stuff/taxonomies/it.txt
```

### Vision Model Fallback

The "Vision fallback" setting allows configuring a secondary vision model when the primary one fails:
- `none`: no fallback (default)
- `auto`: automatically use llava:7b as fallback
- explicit model: e.g., `llava:7b`, `minicpm-v`

## Security & Privacy
- Local-first: the goal is to avoid sending content to external services.
- Files are read and analyzed locally; when OCR is enabled, text is extracted from scans/images too.
- If you switch provider/models, review their policies and how they handle data.

## Limitations (updated over time)
- Parsing and OCR are best-effort: some files may be `skipped` or produce incomplete output.
- There is no “apply plan with per-file approval” workflow yet: moving is manual (`m` / `M`).

## Optional System Dependencies

### OCR for scanned PDFs and images (recommended)
If a PDF has no extractable text (i.e. it's effectively an image), or if you scan documents as images, amenity-ai can use Tesseract OCR.

- Ubuntu / Linux Mint:
  - `sudo apt-get install tesseract-ocr tesseract-ocr-ita`

### `.doc` / `.xls` extraction (optional)
`amenity-ai` can extract text from:
- `.docx` and `.xlsx` without extra dependencies (best-effort)
- `.doc` and `.xls` via LibreOffice (best-effort)

- Ubuntu / Linux Mint:
  - `sudo apt-get install libreoffice`

### `.rtf` extraction (optional)
RTF is supported without dependencies via a naive fallback, but you get better results with `unrtf`.

- Ubuntu / Linux Mint:
  - `sudo apt-get install unrtf`

## LLM Providers

Three providers are supported. You configure one URL each and nothing more — on startup
the app queries them in parallel and discovers which models are available:

| provider | endpoint queried | parallel | notes |
|---|---|---|---|
| `ollama` | `GET /api/tags` | 1 | default `http://localhost:11434`; the only one that can install models |
| `vllm` | `GET /v1/models` | 4 | any vLLM server; empty by default |
| `ds4` | `GET /v1/models` | 1 | any other OpenAI-compatible server (llama.cpp, …); empty by default |

All three may run on another machine on your local network. Leave a URL empty to disable
that provider. Nothing is sent outside the machines you point the app at.

**Parallel** is how many requests a scan sends that provider at once, editable beside each
endpoint in Settings. The default of 4 for vLLM is measured, not guessed: on a 27B model
throughput stops improving past four concurrent requests and per-file latency keeps
growing. ds4 stays at 1 because it answers one caller at a time. Ollama stays at 1 because
`OLLAMA_NUM_PARALLEL` is not visible to us — raise it if your server allows more.

Models carry their provider as a prefix — `ollama:qwen3:8b`, `vllm:…`, `ds4:…` — so you can
always tell where one comes from. With models set to `auto`, candidates are ordered by
provider priority (`vllm` first, since it serves concurrent requests, then `ollama`, then
`ds4`), then by a size band suited to the job: small and fast for extracting facts, mid-size
for classification, small for vision.

Run `amenity-ai doctor` to see what was found and what is missing.

## Scan (MVP)

The table lists all files found in the selected source folder. Unsupported formats are shown as `skipped` with reason `unsupported file type`.

Supported formats include:
- `pdf`
- images: `jpg/jpeg/png`
- office: `doc/docx/odt/xls/xlsx` (see optional dependencies above)
- text: `txt/md/json/rtf/svg/kmz`
- data: `csv/yaml/yml`
- web: `html/htm`
- GPS: `gpx`

See `EXTRACTORS.md` for details on how each format is handled and how to add new ones.

### Keys
- `ctrl+r` reload dir
- `s` scan row (facts extraction, force)
- `S` scan pending (facts extraction)
- `c` classify row (requires `scanned`)
- `C` classify scanned (`scanned` only, per-file)
- `m` move selected eligible file to archive (`classified`, `skipped`, `error`)
- `M` move all eligible files to archive
- `x` stop current task (scan, classify, move)
- `enter` open selected file (default app)
- `u` unclassify selected row (keep scan results)
- `r` reset selected row (back to `pending`, invalidate cache)
- `R` reset all + clear cache (confirmation)
- `F2` settings
- `d` doctor (providers and models)
- `q` or `ctrl+c` quit

During extraction/classification, status transitions and the UI remains interactive while results update row by row.

Mouse text selection is supported (so you can select/copy fields like absolute paths).

### Status
The `Status` column is icon-only (with color):
- `·` pending
- `✓` scanning / classifying / moving (running)
- `✓` scanned (facts available, not yet classified)
- `✓` classified (category/year/name proposed)
- `✓` moved (archived)
- `✗` skipped / error

### Cache (MVP)

Results are cached in `<source>/.amenity-stuff/cache.json` and reused on re-scan.

When you move files to the archive, a separate cache is maintained in `<archive>/.amenity-stuff/cache.json`,
and the source cache entries are kept with status `moved` (including `moved_to`).

An append-only move log is also written to `<archive>/.amenity-stuff/moves.jsonl` (one JSON record per moved file).
- `r` invalidates cache for the selected file
- `R` clears the cache for the whole batch
- `u` keeps scan results but clears classification fields
