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

## Settings

You can change:
- output language (`auto`, `it`, `en`)
- taxonomy (allowed categories)
- models (facts / classify / vision / vision fallback), archive folder, filename separator, OCR mode

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

## LLM Provider

On startup the app tries to detect:
- `ollama` (if available in `PATH`)

Models: the app uses a text model and (for images) a vision model; exact model names are configurable.

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
