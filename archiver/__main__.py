from __future__ import annotations

import argparse
import os
from pathlib import Path

from .app import ArchiverApp
from .config import load_config
from .perf_report import print_performance_report
from .settings import Settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amenity-ai")
    sub = parser.add_subparsers(dest="command")

    # Backward-compatible defaults: allow `amenity-ai` (no subcommand) to run the TUI,
    # and keep --source/--archive available at the top-level.
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("."),
        help="Source folder to analyze (default: .)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("./ARCHIVE"),
        help="Archive root destination (default: ./ARCHIVE)",
    )

    run = sub.add_parser("run", help="Run the TUI (default)")
    run.add_argument(
        "--source",
        type=Path,
        default=Path("."),
        help="Source folder to analyze (default: .)",
    )
    run.add_argument(
        "--archive",
        type=Path,
        default=Path("./ARCHIVE"),
        help="Archive root destination (default: ./ARCHIVE)",
    )

    report = sub.add_parser("report", help="Print a short performance report from the cache")
    report.add_argument(
        "--source",
        type=Path,
        default=Path("."),
        help="Source folder (used to locate <source>/.amenity-stuff/cache.json)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if getattr(args, "command", None) == "report":
        print_performance_report(source_root=args.source)
        return

    # Default command is the TUI run.
    if getattr(args, "command", None) not in {None, "run"}:
        parser.print_help()
        return

    cfg = load_config()
    default_source = args.source
    default_archive = args.archive
    if default_archive == Path("./ARCHIVE") and cfg.last_archive_root:
        default_archive = Path(cfg.last_archive_root)
    skip_setup = bool(cfg.last_archive_root) and args.archive == Path("./ARCHIVE")
    settings = Settings(
        source_root=default_source,
        archive_root=default_archive,
        output_language=cfg.output_language,
        taxonomies=cfg.taxonomies,
        facts_model=cfg.facts_model,
        classify_model=cfg.classify_model,
        vision_model=cfg.vision_model,
        vision_model_fallback=cfg.vision_model_fallback,
        filename_separator=cfg.filename_separator,
        ocr_mode=cfg.ocr_mode,
        undated_folder_name=cfg.undated_folder_name,
        skip_initial_setup=skip_setup,
    )
    # Disable mouse tracking so the terminal can do native text selection (copy with mouse).
    try:
        ArchiverApp(settings).run(mouse=False)
    except KeyboardInterrupt:
        # Textual runs worker threads for long-running tasks (OCR/LLM). A SIGINT while a worker is
        # running may block on thread shutdown. Exit immediately to avoid a noisy traceback.
        os._exit(130)


if __name__ == "__main__":
    main()
