from __future__ import annotations

import asyncio
import sys
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import replace

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import DataTable, Footer, Header, Static
from textual.worker import get_current_worker

from .analyzer import extract_facts_item
from .cache import CacheStore, SaveThrottle
from .concurrency import ConcurrencyLimiter, pool_size_for
from .config import AppConfig, save_config
from .confirm_screen import ConfirmResult, ConfirmScreen
from .discovery import DiscoveryResult, discover_providers
from .normalizer import normalize_items, normalize_items_with_fallback
from .archive_apply import apply_archive_move, archive_dest_for_item
from .scanner import ScanItem, scan_files
from .settings import Settings
from .settings_screen import SettingsResult, SettingsScreen
from .setup_screen import SetupResult, SetupScreen
from .setup_logic import app_config_from_settings, settings_from_setup
from .taxonomy import parse_taxonomy_lines
from .ui_status import (
    RunProgress,
    app_title,
    compute_rate,
    notes_line,
    progress_line,
    provider_summary,
    status_cell,
)
from .model_selection import ROLE_CLASSIFY, rank_models
from .probe_cache import load_probe_cache
from .task_builders import build_analysis_config, _with_pin
from .ui_details import render_details
from .task_state import TaskState
from .help_screen import HelpScreen
from .doctor_screen import DoctorScreen
from .ui_runtime import banner_for_state, count_statuses, derive_task_state, runtime_problem
from .item_mutations import mark_item_classifying, mark_item_scanning, reset_item_to_pending, unclassify_item
from .open_file import open_with_default_app
from .ui_files_table import build_file_table_rows
from .cache_overlay import overlay_scan_items_with_cache


class ArchiverApp(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 4; }
    #banner { height: 1; padding: 0 1; }
    #notes { height: 1; color: $text-muted; padding: 0 1; }
    #files { height: 1fr; }
    #details_box { height: 9; border: round $accent; background: $panel; }
    #details_text { padding: 0 2; }

    #meta { height: 1; }
    #src { width: 1fr; overflow: hidden; }
    #arc { width: 1fr; overflow: hidden; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("s", "extract_row", "Scan", show=True),
        Binding("c", "classify_row", "Classify", show=True),
        Binding("m", "archive_row", "Archive", show=True),
        Binding("u", "unclassify_row", "Unclassify", show=True),
        Binding("r", "reset_row", "Reset", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("f1", "help", "Help", show=True),
        Binding("f2", "settings", "Settings", show=True),
        Binding("d", "doctor", "Doctor", show=True),
        Binding("ctrl+r", "scan", "Reload dir", show=False),
        Binding("S", "extract_pending", "Scan pending", show=False),
        Binding("C", "classify_batch", "Classify scanned", show=False),
        Binding("M", "archive_batch", "Archive eligible", show=False),
        Binding("x", "stop_analysis", "Stop analysis", show=False),
        Binding("enter", "open_file", "Open file", show=False),
        Binding("return", "open_file", "Open file", show=False),
        Binding("R", "reset_all", "Reset all", show=False),
        Binding("U", "unclassify_all", "Unclassify all", show=False),
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.title = app_title()
        self._discovery: DiscoveryResult | None = None
        self._scan_items: list[ScanItem] = []
        self._scan_index_by_path: dict[str, int] = {}
        self._analysis_task = TaskState()
        self._cache: CacheStore | None = None
        self._scan_task = TaskState()
        self._archive_task = TaskState()
        self._provider_line: str = ""
        # Progress of the run in flight, not of the table: the table can hold
        # files finished in an earlier session.
        self._run_total = 0
        self._run_completed = 0
        self._run_in_flight = 0
        self._run_skipped = 0
        self._run_error = 0
        self._run_completions: list[float] = []
        self._run_started_at = 0.0
        self._cache_throttle = SaveThrottle()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Container(id="top"):
            yield Static(f"Source: {self.settings.source_root}", id="src")
            yield Static(f"Archive: {self.settings.archive_root}", id="arc")
            yield Static("Ready.", id="notes")
            yield Static("", id="banner")

        files = DataTable(id="files")
        files.add_column(" ", key="status", width=2)
        files.add_column("Type", key="kind")
        files.add_column("File", key="file")
        files.add_column("Category", key="category")
        files.add_column("Year", key="year")
        files.cursor_type = "row"
        yield files

        with Container(id="details_box"):
            yield Static("", id="details_text")

        yield Footer()

    async def on_mount(self) -> None:
        initial_source = self.settings.source_root.expanduser().resolve()
        initial_archive = self.settings.archive_root.expanduser().resolve()
        if self.settings.skip_initial_setup:
            self._apply_setup(setup=SetupResult(source_root=initial_source, archive_root=initial_archive))
            asyncio.create_task(self._post_setup())
            return
        self.push_screen(
            SetupScreen(source_root=initial_source, archive_root=initial_archive),
            callback=self._on_setup_done,
            wait_for_dismiss=False,
        )

    def _on_setup_done(self, setup: SetupResult) -> None:
        self._apply_setup(setup=setup)
        asyncio.create_task(self._post_setup())

    def _apply_setup(self, *, setup: SetupResult) -> None:
        self.settings = settings_from_setup(current=self.settings, setup=setup)
        self.query_one("#src", Static).update(f"Source: {self.settings.source_root}")
        self.query_one("#arc", Static).update(f"Archive: {self.settings.archive_root}")
        self._cache = CacheStore(self.settings.source_root)
        self._cache.load()
        self._save_app_config()

    def _save_app_config(self) -> None:
        save_config(app_config_from_settings(self.settings))

    async def _post_setup(self) -> None:
        await self._run_discovery()
        await self._run_scan()
        self.query_one("#files", DataTable).focus()
        self._update_details_from_cursor()

    async def action_scan(self) -> None:
        await self._run_scan()

    async def action_extract_row(self) -> None:
        await self._run_extract_row(force=True)

    async def action_extract_pending(self) -> None:
        await self._run_extract_pending()

    async def action_classify_row(self) -> None:
        await self._run_classify_row(force=True)

    async def action_classify_batch(self) -> None:
        await self._run_classify_batch()

    async def action_archive_row(self) -> None:
        await self._run_archive_row()

    async def action_archive_batch(self) -> None:
        await self._run_archive_batch()

    async def action_open_file(self) -> None:
        files = self.query_one("#files", DataTable)
        row_index = files.cursor_row
        if row_index < 0 or row_index >= len(self._scan_items):
            return
        path = self._scan_items[row_index].path
        open_with_default_app(path)

    # Backward-compatible actions (no longer bound to keys).
    async def action_analyze_row(self) -> None:
        await self.action_extract_row()

    async def action_analyze_pending(self) -> None:
        await self.action_extract_pending()

    async def action_normalize_ready(self) -> None:
        await self.action_classify_batch()

    async def action_stop_analysis(self) -> None:
        stopped_any = False
        if self._analysis_task.running:
            self._analysis_task.request_cancel()
            stopped_any = True
        if self._scan_task.running:
            self._scan_task.request_cancel()
            stopped_any = True
        if self._archive_task.running:
            self._archive_task.request_cancel()
            stopped_any = True
        if stopped_any:
            self._render_notes()

    async def action_reset_row(self) -> None:
        if self._analysis_task.running or self._scan_task.running or self._archive_task.running:
            return
        files = self.query_one("#files", DataTable)
        row_index = files.cursor_row
        if row_index < 0 or row_index >= len(self._scan_items):
            return
        item = self._scan_items[row_index]
        key = str(item.path)
        reset = reset_item_to_pending(item)
        self._scan_items[row_index] = reset
        if self._cache:
            self._cache.invalidate(item)
            self._cache.save()
        files.update_cell(key, "status", status_cell("pending"))
        files.update_cell(key, "category", "")
        files.update_cell(key, "year", "")
        self._update_details(row_index)
        self._render_notes()

    async def action_reset_all(self) -> None:
        if self._analysis_task.running or self._scan_task.running or self._archive_task.running:
            return
        self.push_screen(
            ConfirmScreen(message="Reset ALL files and clear cache?"),
            callback=self._on_reset_all_confirmed,
            wait_for_dismiss=False,
        )

    def _on_reset_all_confirmed(self, result: ConfirmResult) -> None:
        if not result.confirmed:
            return
        self._reset_all_impl()

    def _reset_all_impl(self) -> None:
        if self._cache:
            self._cache.clear()
            self._cache.save()
        self._scan_items = [reset_item_to_pending(it) for it in self._scan_items]
        self._render_files()
        self._update_details_from_cursor()
        self._render_notes()

    async def action_unclassify_row(self) -> None:
        if self._analysis_task.running or self._scan_task.running or self._archive_task.running:
            return
        files = self.query_one("#files", DataTable)
        row_index = files.cursor_row
        if row_index < 0 or row_index >= len(self._scan_items):
            return
        item = self._scan_items[row_index]
        if item.status != "classified":
            return
        key = str(item.path)
        updated = unclassify_item(item)
        self._scan_items[row_index] = updated
        if self._cache:
            self._cache.upsert(updated)
            self._cache.save()
        files.update_cell(key, "status", status_cell("scanned"))
        files.update_cell(key, "category", "")
        files.update_cell(key, "year", "")
        self._update_details(row_index)
        self._render_notes()

    async def action_unclassify_all(self) -> None:
        if self._analysis_task.running or self._scan_task.running or self._archive_task.running:
            return
        self.push_screen(
            ConfirmScreen(message="Unclassify ALL classified files (keep scan results)?"),
            callback=self._on_unclassify_all_confirmed,
            wait_for_dismiss=False,
        )

    def _on_unclassify_all_confirmed(self, result: ConfirmResult) -> None:
        if not result.confirmed:
            return
        self._unclassify_all_impl()

    def _unclassify_all_impl(self) -> None:
        files = self.query_one("#files", DataTable)
        any_changed = False
        for idx, it in enumerate(list(self._scan_items)):
            if it.status != "classified":
                continue
            any_changed = True
            updated = unclassify_item(it)
            self._scan_items[idx] = updated
            key = str(updated.path)
            files.update_cell(key, "status", status_cell("scanned"))
            files.update_cell(key, "category", "")
            files.update_cell(key, "year", "")
            if self._cache:
                self._cache.upsert(updated)
        if any_changed and self._cache:
            self._cache.save()
        self._update_details_from_cursor()
        self._render_notes()

    async def action_settings(self) -> None:
        if self._analysis_task.running or self._archive_task.running:
            return
        if self._scan_task.running:
            return
        available_models = self._discovery.models if self._discovery else ()
        provider_info = "Provider: (not detected yet)"
        if self._discovery:
            provider_info = self._provider_line or "Provider: none reachable • models: 0"
            if available_models:
                ids = [m.id for m in available_models]
                shown = ", ".join(ids[:8])
                if len(ids) > 8:
                    shown += f" (+{len(ids) - 8})"
                provider_info = f"{provider_info}\nModels: {shown}"
        self.push_screen(
            SettingsScreen(
                output_language=self.settings.output_language,
                taxonomies=self.settings.taxonomies,
                facts_model=self.settings.facts_model,
                classify_model=self.settings.classify_model,
                vision_model=self.settings.vision_model,
                filename_separator=self.settings.filename_separator,
                ocr_mode=self.settings.ocr_mode,
                undated_folder_name=self.settings.undated_folder_name,
                archive_root=self.settings.archive_root,
                providers=self.settings.providers,
                provider_concurrency=self.settings.provider_concurrency,
                available_models=available_models,
                provider_info=provider_info,
            ),
            callback=self._on_settings_done,
            wait_for_dismiss=False,
        )

    async def action_doctor(self) -> None:
        if self._analysis_task.running or self._archive_task.running or self._scan_task.running:
            return
        self.push_screen(
            DoctorScreen(
                settings=self.settings,
                discovery=self._discovery,
                on_refresh=lambda: self.run_worker(self._run_discovery()),
            ),
            wait_for_dismiss=False,
        )

    async def action_help(self) -> None:
        self.push_screen(HelpScreen(), wait_for_dismiss=False)

    def _on_settings_done(self, result: SettingsResult) -> None:
        endpoints_changed = result.providers != self.settings.providers
        self.settings = replace(
            self.settings,
            output_language=result.output_language,
            taxonomies=result.taxonomies,
            facts_model=result.facts_model,
            classify_model=result.classify_model,
            vision_model=result.vision_model,
            archive_root=result.archive_root,
            filename_separator=result.filename_separator,
            ocr_mode=result.ocr_mode,
            undated_folder_name=result.undated_folder_name,
            providers=result.providers,
            provider_concurrency=result.provider_concurrency,
        )
        self.query_one("#arc", Static).update(f"Archive: {self.settings.archive_root}")
        self._save_app_config()
        self._render_notes()
        if endpoints_changed:
            self.run_worker(self._run_discovery())

    async def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "files":
            return
        self._update_details(event.cursor_row)

    async def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        if event.data_table.id != "files":
            return
        self._update_details(event.coordinate.row)

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "files":
            return
        await self.action_open_file()

    async def _run_discovery(self) -> None:
        notes_widget = self.query_one("#notes", Static)
        notes_widget.update("Detecting local providers…")

        def do_discover() -> DiscoveryResult:
            return discover_providers(
                self.settings.providers, probe_cache=load_probe_cache()
            )

        worker = self.run_worker(do_discover, thread=True)
        self._discovery = await worker.wait()
        self._provider_line = provider_summary(self._discovery, self.settings)
        self.title = app_title()
        self._render_notes()

    async def _run_scan(self) -> None:
        if self._analysis_task.running or self._archive_task.running:
            return
        notes_widget = self.query_one("#notes", Static)
        notes_widget.update("Scanning files…")

        files = self.query_one("#files", DataTable)
        files.clear()

        def do_scan() -> list[ScanItem]:
            worker = get_current_worker()
            return scan_files(
                self.settings.source_root,
                recursive=self.settings.recursive,
                include_extensions=self.settings.include_extensions,
                exclude_dirnames=self.settings.exclude_dirnames,
                should_cancel=lambda: worker.is_cancelled,
            )

        worker = self.run_worker(do_scan, thread=True, exclusive=True)
        self._scan_task.start(worker)
        self._render_notes()
        self._scan_items = await worker.wait()
        self._scan_task.finish()
        if self._cache:
            self._scan_items = overlay_scan_items_with_cache(self._scan_items, self._cache)
        # Sort by file type: images first (use moondream), then rest (use text LLM only).
        # This minimizes model switching in Ollama.
        self._scan_items = sorted(
            self._scan_items,
            key=lambda it: (0 if it.kind == "image" else 1, it.path.name.lower()),
        )
        self._render_files()
        self._render_notes()
        self._update_details_from_cursor()

    async def _run_extract_pending(self) -> None:
        if self._analysis_task.running:
            return
        if self._scan_task.running:
            return
        if self._archive_task.running:
            return
        self._analysis_task.cancel_requested = False
        self._analysis_task.running = True
        self._render_notes()

        files = self.query_one("#files", DataTable)

        def begin_run(total: int) -> None:
            self._run_total = total
            self._run_completed = 0
            self._run_in_flight = 0
            self._run_skipped = 0
            self._run_error = 0
            self._run_completions.clear()
            self._run_started_at = time.monotonic()
            self._cache_throttle = SaveThrottle()
            self._render_notes()

        def mark_scanning(path_str: str) -> None:
            idx = self._scan_index_by_path.get(path_str)
            if idx is None:
                return
            it = self._scan_items[idx]
            if it.status != "pending":
                return
            self._scan_items[idx] = mark_item_scanning(it)
            self._run_in_flight += 1
            files.update_cell(path_str, "status", status_cell("scanning"))
            self._render_notes()
            if files.cursor_row == idx:
                self._update_details(idx)

        def apply_result(path_str: str, new_item: ScanItem) -> None:
            idx = self._scan_index_by_path.get(path_str)
            if idx is None:
                return
            self._scan_items[idx] = new_item
            self._run_in_flight = max(0, self._run_in_flight - 1)
            self._run_completed += 1
            if new_item.status == "skipped":
                self._run_skipped += 1
            elif new_item.status == "error":
                self._run_error += 1
            self._run_completions.append(time.monotonic())
            files.update_cell(path_str, "status", status_cell(new_item.status))
            files.update_cell(path_str, "category", new_item.category or "")
            files.update_cell(path_str, "year", new_item.reference_year or "")
            self._render_notes()
            if files.cursor_row == idx:
                self._update_details(idx)
            if self._cache:
                self._cache.upsert(new_item)
                # A write re-serialises the whole cache; with several files in
                # flight, doing it per file would stall the event loop.
                if self._cache_throttle.record():
                    self._cache.save()

        def finish(cancelled: bool) -> None:
            if cancelled:
                for idx, it in enumerate(list(self._scan_items)):
                    if it.status != "scanning":
                        continue
                    updated = replace(it, status="pending", reason="Scan stopped")
                    self._scan_items[idx] = updated
                    key = str(updated.path)
                    files.update_cell(key, "status", status_cell("pending"))
            if self._cache and self._cache_throttle.flush():
                self._cache.save()
            self._run_in_flight = 0
            # Also retire the run's total: the progress line is gated on it, and
            # a later task in the same session (a classify, a single-row scan)
            # would otherwise render this run's frozen numbers as its own.
            self._run_total = 0
            self._analysis_task.running = False
            self._render_notes()

        def do_extract_background() -> None:
            worker = get_current_worker()
            taxonomy, _ = parse_taxonomy_lines(self.settings.get_taxonomy_lines())
            limiter = ConcurrencyLimiter.from_limits(self.settings.provider_concurrency)
            cfg = build_analysis_config(
                settings=self.settings, discovery=self._discovery,
                taxonomy=taxonomy, limiter=limiter,
            )
            targets = [it for it in self._scan_items if it.status == "pending"]
            self.call_from_thread(begin_run, len(targets))

            def run_one(it: ScanItem):
                # Checked here as well as in the loop below: a thread that has
                # just finished an item would otherwise pick up the next one
                # before the executor is told to stop.
                if worker.is_cancelled:
                    return None
                path_str = str(it.path)
                self.call_from_thread(mark_scanning, path_str)
                t0 = time.perf_counter()
                try:
                    res = extract_facts_item(it, config=cfg)
                except Exception as exc:  # noqa: BLE001
                    # One bad file must not take the run down with it: an
                    # exception escaping here would leave finish() uncalled and
                    # the UI stuck on "running" forever.
                    return path_str, replace(
                        it, status="error", reason=f"Scan crashed: {type(exc).__name__}"
                    )
                elapsed = time.perf_counter() - t0
                return path_str, replace(
                    it,
                    status=res.status,
                    reason=res.reason,
                    confidence=res.confidence,
                    analysis_time_s=elapsed,
                    model_used=res.model_used,
                    summary_long=res.summary_long,
                    facts_json=res.facts_json,
                    llm_raw_output=res.llm_raw_output,
                    extract_method=res.extract_method,
                    extract_time_s=res.extract_time_s,
                    llm_time_s=res.llm_time_s,
                    ocr_time_s=res.ocr_time_s,
                    ocr_mode=res.ocr_mode,
                    facts_time_s=elapsed,
                    facts_llm_time_s=res.llm_time_s,
                    facts_model_used=res.model_used,
                    category=None,
                    reference_year=None,
                    proposed_name=None,
                    summary=None,
                )

            executor = ThreadPoolExecutor(
                max_workers=pool_size_for(self.settings.providers, limiter),
                thread_name_prefix="facts",
            )
            stopped = False
            try:
                futures = [executor.submit(run_one, it) for it in targets]
                for future in as_completed(futures):
                    if worker.is_cancelled and not stopped:
                        stopped = True
                        # Drop what is queued; what is already running finishes
                        # and is still applied, so no completed work is wasted.
                        executor.shutdown(wait=False, cancel_futures=True)
                    try:
                        outcome = future.result()
                    except CancelledError:
                        continue
                    except Exception:  # noqa: BLE001
                        continue
                    if outcome is None:
                        continue
                    path_str, updated = outcome
                    self.call_from_thread(apply_result, path_str, updated)
            finally:
                executor.shutdown(wait=True)
            self.call_from_thread(finish, worker.is_cancelled)

        worker = self.run_worker(do_extract_background, thread=True, exclusive=True)
        self._analysis_task.worker = worker

    async def _run_classify_batch(self) -> None:
        if self._analysis_task.running or self._scan_task.running or self._archive_task.running:
            return
        if not self._discovery:
            return
        taxonomy, _ = parse_taxonomy_lines(self.settings.get_taxonomy_lines())
        ranked = rank_models(self._discovery.models if self._discovery else (), ROLE_CLASSIFY)
        text_models = _with_pin(ranked, self.settings.classify_model)
        model = text_models[0] if text_models else ""

        targets = [it for it in self._scan_items if it.status == "scanned"]
        if not targets:
            return

        self._analysis_task.cancel_requested = False
        self._analysis_task.running = True
        self._render_notes()

        files = self.query_one("#files", DataTable)
        for it in targets:
            path_str = str(it.path)
            idx = self._scan_index_by_path.get(path_str)
            if idx is None:
                continue
            self._scan_items[idx] = mark_item_classifying(self._scan_items[idx])
            files.update_cell(path_str, "status", status_cell("classifying"))
        self._render_notes()

        def apply_result(path_str: str, updated: ScanItem) -> None:
            idx = self._scan_index_by_path.get(path_str)
            if idx is None:
                return
            self._scan_items[idx] = updated
            files.update_cell(path_str, "status", status_cell(updated.status))
            files.update_cell(path_str, "category", updated.category or "")
            files.update_cell(path_str, "year", updated.reference_year or "")
            self._render_notes()
            if files.cursor_row == idx:
                self._update_details(idx)

        def finish() -> None:
            if self._cache:
                for item in self._scan_items:
                    if item.status in {"classified", "scanned"}:
                        self._cache.upsert(item)
                self._cache.save()
            self._analysis_task.running = False
            self._render_notes()

        def do_classify_background() -> None:
            worker = get_current_worker()
            try:
                batch_size = 12 if len(targets) > 12 else max(4, len(targets))
                t0 = time.perf_counter()
                res = normalize_items_with_fallback(
                    items=targets,
                    models=text_models,
                    provider_urls=self.settings.providers,
                    taxonomy=taxonomy,
                    output_language=self.settings.output_language,
                    filename_separator=self.settings.filename_separator,
                    chunk_size=batch_size,
                    should_cancel=lambda: worker.is_cancelled,
                )
                llm_elapsed = time.perf_counter() - t0

                base_reason = None
                if worker.is_cancelled or res.error == "Cancelled":
                    base_reason = "Classification stopped"
                elif res.error:
                    base_reason = f"Classification error: {res.error}"

                for it in targets:
                    path_str = str(it.path)
                    idx = self._scan_index_by_path.get(path_str)
                    if idx is None:
                        continue
                    cur = self._scan_items[idx]
                    if cur.status != "classifying":
                        continue
                    upd = res.by_path.get(path_str) if res.by_path else None
                    if upd:
                        updated = replace(
                            cur,
                            status="classified",
                            category=upd.get("category") or cur.category or "unknown",
                            reference_year=upd.get("reference_year") or cur.reference_year,
                            proposed_name=upd.get("proposed_name") or cur.proposed_name,
                            summary=upd.get("summary") or cur.summary,
                            model_used=str(upd.get("model_used") or cur.model_used or model),
                            reason=None,
                            classify_time_s=llm_elapsed,
                            classify_llm_time_s=llm_elapsed,
                            classify_model_used=str(upd.get("model_used") or cur.model_used or model),
                        )
                    else:
                        updated = replace(
                            cur,
                            status="scanned",
                            reason=base_reason or "Classification error: no output",
                        )
                    self.call_from_thread(apply_result, path_str, updated)
                self.call_from_thread(finish)
            except Exception as exc:  # noqa: BLE001
                for it in targets:
                    path_str = str(it.path)
                    idx = self._scan_index_by_path.get(path_str)
                    if idx is None:
                        continue
                    cur = self._scan_items[idx]
                    if cur.status != "classifying":
                        continue
                    self.call_from_thread(
                        apply_result,
                        path_str,
                        replace(cur, status="scanned", reason=f"Classification crashed: {type(exc).__name__}"),
                    )
                self.call_from_thread(finish)

        worker = self.run_worker(do_classify_background, thread=True, exclusive=True)
        self._analysis_task.worker = worker

    async def _run_archive_row(self) -> None:
        if self._analysis_task.running or self._scan_task.running or self._archive_task.running:
            return
        files = self.query_one("#files", DataTable)
        row_index = files.cursor_row
        if row_index < 0 or row_index >= len(self._scan_items):
            return
        it = self._scan_items[row_index]
        if it.status not in {"classified", "skipped", "error"}:
            return
        await self._run_archive_targets([str(it.path)])

    async def _run_archive_batch(self) -> None:
        if self._analysis_task.running or self._scan_task.running or self._archive_task.running:
            return
        targets = [it for it in self._scan_items if it.status in {"classified", "skipped", "error"}]
        if not targets:
            return
        preview = [str(it.path) for it in targets[:3]]
        more = "" if len(targets) <= 3 else f"\n… +{len(targets) - 3} more"
        msg = "\n".join(
            [
                f"Move {len(targets)} eligible files to archive?",
                "",
                *[f"- {p}" for p in preview],
                more,
            ]
        ).rstrip()

        def after_confirm(result: ConfirmResult) -> None:
            if not result.confirmed:
                return
            asyncio.create_task(self._run_archive_targets([str(it.path) for it in targets]))

        self.push_screen(ConfirmScreen(message=msg), callback=after_confirm, wait_for_dismiss=False)

    async def _run_archive_targets(self, keys: list[str]) -> None:
        if self._archive_task.running:
            return
        files = self.query_one("#files", DataTable)

        source_cache = self._cache
        archive_cache = CacheStore(self.settings.archive_root)
        archive_cache.load()

        self._archive_task.cancel_requested = False
        self._archive_task.running = True
        self._render_notes()

        prev_status_by_path: dict[str, str] = {}

        def mark_moving(path_str: str, prev_status: str) -> None:
            idx = self._scan_index_by_path.get(path_str)
            if idx is None:
                return
            it = self._scan_items[idx]
            if it.status not in {"classified", "skipped", "error"}:
                return
            prev_status_by_path[path_str] = prev_status
            self._scan_items[idx] = replace(it, status="moving", reason=None)
            files.update_cell(path_str, "status", status_cell("moving"))
            self._render_notes()
            if files.cursor_row == idx:
                self._update_details(idx)

        def apply_result(path_str: str, updated: ScanItem) -> None:
            idx = self._scan_index_by_path.get(path_str)
            if idx is None:
                return
            self._scan_items[idx] = updated
            files.update_cell(path_str, "status", status_cell(updated.status))
            self._render_notes()
            if files.cursor_row == idx:
                self._update_details(idx)
            prev_status_by_path.pop(path_str, None)

        def finish(cancelled: bool) -> None:
            if cancelled:
                for idx, it in enumerate(list(self._scan_items)):
                    if it.status != "moving":
                        continue
                    prev = prev_status_by_path.get(str(it.path), "classified")
                    updated = replace(it, status=prev, reason="Move stopped")
                    self._scan_items[idx] = updated
                    key = str(updated.path)
                    files.update_cell(key, "status", status_cell(updated.status))
            self._archive_task.running = False
            self._render_notes()

        def do_move_background() -> None:
            worker = get_current_worker()
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            for path_str in keys:
                if worker.is_cancelled:
                    break
                idx = self._scan_index_by_path.get(path_str)
                if idx is None:
                    continue
                it = self._scan_items[idx]
                if it.status not in {"classified", "skipped", "error"}:
                    continue
                self.call_from_thread(mark_moving, path_str, it.status)
                try:
                    updated, _dest_rel = apply_archive_move(
                        it,
                        settings=self.settings,
                        source_cache=source_cache,
                        archive_cache=archive_cache,
                        now_iso=now_iso,
                    )
                except Exception as exc:  # noqa: BLE001
                    updated = replace(it, status="error", reason=f"Move failed: {type(exc).__name__}")
                    if source_cache:
                        source_cache.upsert(updated)
                        source_cache.save()
                self.call_from_thread(apply_result, path_str, updated)
            self.call_from_thread(finish, worker.is_cancelled)

        worker = self.run_worker(do_move_background, thread=True, exclusive=True)
        self._archive_task.worker = worker

    async def _run_extract_row(self, *, force: bool) -> None:
        if self._analysis_task.running or self._scan_task.running or self._archive_task.running:
            return
        files = self.query_one("#files", DataTable)
        row_index = files.cursor_row
        if row_index < 0 or row_index >= len(self._scan_items):
            return
        it = self._scan_items[row_index]
        if self._cache:
            self._cache.invalidate(it)
            self._cache.save()
        reset = reset_item_to_pending(it)
        if not force:
            reset = replace(reset, status=it.status)
        self._scan_items[row_index] = reset
        path_str = str(reset.path)
        mark_item = mark_item_scanning(reset)
        self._scan_items[row_index] = mark_item
        files.update_cell(path_str, "status", status_cell("scanning"))
        self._update_details(row_index)
        self._analysis_task.cancel_requested = False
        self._analysis_task.running = True
        self._render_notes()

        def do_one() -> None:
            worker = get_current_worker()

            def finish_with_item(item: ScanItem) -> None:
                idx = self._scan_index_by_path.get(path_str)
                if idx is None:
                    self._analysis_task.running = False
                    self._render_notes()
                    return
                self._scan_items[idx] = item
                files.update_cell(path_str, "status", status_cell(item.status))
                files.update_cell(path_str, "category", item.category or "")
                files.update_cell(path_str, "year", item.reference_year or "")
                self._update_details(idx)
                if self._cache and item.status not in ("pending", "scanning"):
                    self._cache.upsert(item)
                    self._cache.save()
                self._analysis_task.running = False
                self._render_notes()

            taxonomy, _ = parse_taxonomy_lines(self.settings.get_taxonomy_lines())
            # One file on its own still respects a provider that wants a single
            # caller — nothing else is in flight here, but ds4 does not know that.
            cfg = build_analysis_config(
                settings=self.settings,
                discovery=self._discovery,
                taxonomy=taxonomy,
                limiter=ConcurrencyLimiter.from_limits(self.settings.provider_concurrency),
            )
            t0 = time.perf_counter()
            if worker.is_cancelled:
                stopped = replace(reset, status="pending", reason="Scan stopped")
                self.call_from_thread(finish_with_item, stopped)
                return
            res = extract_facts_item(reset, config=cfg)
            elapsed = time.perf_counter() - t0
            if worker.is_cancelled:
                stopped = replace(reset, status="pending", reason="Scan stopped")
                self.call_from_thread(finish_with_item, stopped)
                return
            updated = replace(
                reset,
                status=res.status,
                reason=res.reason,
                confidence=res.confidence,
                analysis_time_s=elapsed,
                model_used=res.model_used,
                summary_long=res.summary_long,
                facts_json=res.facts_json,
                llm_raw_output=res.llm_raw_output,
                extract_method=res.extract_method,
                extract_time_s=res.extract_time_s,
                llm_time_s=res.llm_time_s,
                ocr_time_s=res.ocr_time_s,
                ocr_mode=res.ocr_mode,
                facts_time_s=elapsed,
                facts_llm_time_s=res.llm_time_s,
                facts_model_used=res.model_used,
                category=None,
                reference_year=None,
                proposed_name=None,
                summary=None,
            )

            self.call_from_thread(finish_with_item, updated)

        worker = self.run_worker(do_one, thread=True, exclusive=True)
        self._analysis_task.worker = worker

    async def _run_classify_row(self, *, force: bool) -> None:
        if self._analysis_task.running or self._scan_task.running or self._archive_task.running:
            return
        if not self._discovery:
            return
        files = self.query_one("#files", DataTable)
        row_index = files.cursor_row
        if row_index < 0 or row_index >= len(self._scan_items):
            return
        it = self._scan_items[row_index]
        if it.status != "scanned":
            self.query_one("#notes", Static).update("Select a scanned file first (press S/s).")
            return

        taxonomy, _ = parse_taxonomy_lines(self.settings.get_taxonomy_lines())
        ranked = rank_models(self._discovery.models if self._discovery else (), ROLE_CLASSIFY)
        text_models = _with_pin(ranked, self.settings.classify_model)
        model = text_models[0] if text_models else ""

        key = str(it.path)
        self._scan_items[row_index] = mark_item_classifying(it)
        files.update_cell(key, "status", status_cell("classifying"))
        self._update_details(row_index)
        self._analysis_task.cancel_requested = False
        self._analysis_task.running = True
        self._render_notes()

        def do_one() -> None:
            worker = get_current_worker()

            def finish_with_item(item: ScanItem) -> None:
                idx = self._scan_index_by_path.get(key)
                if idx is None:
                    self._analysis_task.running = False
                    self._render_notes()
                    return
                self._scan_items[idx] = item
                files.update_cell(key, "status", status_cell(item.status))
                files.update_cell(key, "category", item.category or "")
                files.update_cell(key, "year", item.reference_year or "")
                self._update_details(idx)
                if self._cache and item.status not in ("scanned", "classifying"):
                    self._cache.upsert(item)
                    self._cache.save()
                self._analysis_task.running = False
                self._render_notes()

            t0 = time.perf_counter()
            if worker.is_cancelled:
                stopped = replace(it, status="scanned", reason="Classification stopped")
                self.call_from_thread(finish_with_item, stopped)
                return
            res = normalize_items_with_fallback(
                items=[it],
                models=text_models,
                provider_urls=self.settings.providers,
                taxonomy=taxonomy,
                output_language=self.settings.output_language,
                filename_separator=self.settings.filename_separator,
                chunk_size=1,
                should_cancel=lambda: worker.is_cancelled,
            )
            llm_elapsed = time.perf_counter() - t0
            if worker.is_cancelled:
                stopped = replace(it, status="scanned", reason="Classification stopped")
                self.call_from_thread(finish_with_item, stopped)
                return
            upd = res.by_path.get(key) if res.by_path else None
            if res.error or not upd:
                updated = replace(it, status="scanned", reason=f"Classification error: {res.error or 'no output'}")
            else:
                updated = replace(
                    it,
                    status="classified",
                    reason=None,
                    category=upd.get("category") or "unknown",
                    reference_year=upd.get("reference_year") or None,
                    proposed_name=upd.get("proposed_name") or it.proposed_name,
                    summary=upd.get("summary") or it.summary,
                    confidence=upd.get("confidence") if isinstance(upd.get("confidence"), (int, float)) else it.confidence,
                    model_used=str(upd.get("model_used") or model),
                    classify_time_s=llm_elapsed,
                    classify_llm_time_s=llm_elapsed,
                    classify_model_used=str(upd.get("model_used") or model),
                )

            self.call_from_thread(finish_with_item, updated)

        worker = self.run_worker(do_one, thread=True, exclusive=True)
        self._analysis_task.worker = worker

    def _render_files(self) -> None:
        files = self.query_one("#files", DataTable)
        prev_row = files.cursor_row
        files.clear()
        self._scan_index_by_path.clear()

        rows, index_by_key = build_file_table_rows(self._scan_items, source_root=self.settings.source_root)
        self._scan_index_by_path.update(index_by_key)
        for row in rows:
            files.add_row(row.status, row.kind, row.file, row.category, row.year, key=row.key)

        if files.row_count:
            if prev_row < 0 or prev_row >= files.row_count:
                files.move_cursor(row=0, column=0, scroll=False)
            else:
                files.move_cursor(row=prev_row, column=0, scroll=False)

    def _update_details_from_cursor(self) -> None:
        table = self.query_one("#files", DataTable)
        if table.row_count == 0:
            self.query_one("#details_text", Static).update("")
            return
        self._update_details(table.cursor_row)

    def _update_details(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._scan_items):
            return
        item = self._scan_items[row_index]
        details_widget = self.query_one("#details_text", Static)
        width = details_widget.size.width or (self.size.width - 4)
        details_widget.update(
            # Let the widget clip to the fixed panel height; avoid adding our own “…” line.
            render_details(item, settings=self.settings, max_width=max(40, width))
        )

    def _render_notes(self) -> None:
        counts = count_statuses(self._scan_items)
        state = derive_task_state(counts=counts, analysis=self._analysis_task, scan=self._scan_task, archive=self._archive_task)

        problem, severity = runtime_problem(self._discovery, self.settings)
        banner_text, banner_style = banner_for_state(
            state=state,
            scanning=counts.scanning,
            classifying=counts.classifying,
            moving=counts.moving,
            problem=problem,
            severity=severity,
        )
        if self._analysis_task.running and self._run_total:
            text = progress_line(
                RunProgress(
                    total=self._run_total,
                    completed=self._run_completed,
                    in_flight=self._run_in_flight,
                    skipped=self._run_skipped,
                    error=self._run_error,
                ),
                rate=compute_rate(
                    self._run_completions,
                    now=time.monotonic(),
                    started_at=self._run_started_at,
                ),
                total_files=counts.total,
            )
        else:
            text = notes_line(
                scan_items_total=counts.total,
                pending=counts.pending,
                scanning=counts.scanning,
                scanned=counts.scanned,
                classifying=counts.classifying,
                classified=counts.classified,
                moved=counts.moved,
                skipped=counts.skipped,
                error=counts.error,
            )
        self.query_one("#notes", Static).update(text)
        self.query_one("#banner", Static).update(Text(banner_text, style=banner_style))
