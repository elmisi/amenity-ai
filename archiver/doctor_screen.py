"""The doctor screen: shows the report and knows how to act on it.

No diagnosis logic lives here: run_doctor decides, this class renders. The
probe and the download run in workers because they must never block
Textual's event loop.
"""
from __future__ import annotations

from typing import Callable, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, OptionList, Static

from .capabilities import CAP_COMPLETION, CAP_VISION
from .discovery import discover_providers
from .doctor import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIP,
    STATUS_WARN,
    DoctorReport,
    Remedy,
    run_doctor,
)
from .ollama_admin import PullProgress, probe_vision, pull_model
from .probe_cache import load_probe_cache, save_probe_result

_ICON = {STATUS_OK: "✓", STATUS_WARN: "⚠", STATUS_FAIL: "✗", STATUS_SKIP: "—"}
_GB = 1024 ** 3


def _network_probe(*, url: str, bare_id: str) -> Optional[bool]:
    """Query the model, and remember only the conclusive answers."""
    verdict = probe_vision(base_url=url, model=bare_id)
    if verdict is not None:
        caps = {CAP_COMPLETION} | ({CAP_VISION} if verdict else set())
        save_probe_result(url=url, bare_id=bare_id, capabilities=frozenset(caps))
    return verdict


class DoctorScreen(ModalScreen[None]):
    CSS = """
    DoctorScreen { layout: vertical; }
    #intro { height: auto; color: $text-muted; }
    #report { height: auto; border: round $accent; background: $panel; padding: 1 2; }
    #remedies { height: auto; border: round $accent; background: $panel; }
    #pull_status { height: auto; padding: 1 2; }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
        Binding("r", "refresh", "Refresh"),
        Binding("x", "cancel_pull", "Cancel download"),
    ]

    def __init__(
        self,
        *,
        settings,
        discovery=None,
        on_refresh: Optional[Callable[[], None]] = None,
        on_report: Optional[Callable[[DoctorReport], None]] = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._discovery = discovery
        self._on_refresh = on_refresh
        self._on_report = on_report
        self._report: Optional[DoctorReport] = None
        self._remedies: tuple[Remedy, ...] = ()
        self._cancel_pull = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "Doctor: r re-check • Enter install • x cancel • Esc close",
            id="intro",
        )
        yield Static("Checking…", id="report", markup=False)
        yield OptionList(id="remedies")
        yield Static("", id="pull_status", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh()

    # --- diagnosis ------------------------------------------------------

    def action_refresh(self) -> None:
        self.query_one("#pull_status", Static).update("")
        self.run_worker(self._diagnose, thread=True, exclusive=True)

    def _diagnose(self) -> None:
        discovery = discover_providers(
            self._settings.providers,
            probe_cache=load_probe_cache(),
            disabled=self._settings.disabled_providers(),
        )
        report = run_doctor(
            discovery=discovery, settings=self._settings, probe=_network_probe
        )
        self.app.call_from_thread(self._show_report, discovery, report)

    def _show_report(self, discovery, report: DoctorReport) -> None:
        self._discovery = discovery
        self._report = report
        if self._on_report is not None:
            self._on_report(report)

        lines = [
            f"{_ICON.get(check.status, '?')} {check.label} — {check.detail}"
            for check in report.checks
        ]
        self.query_one("#report", Static).update("\n".join(lines))

        self._remedies = tuple(r for check in report.checks for r in check.remedies)
        option_list = self.query_one("#remedies", OptionList)
        option_list.clear_options()
        for remedy in self._remedies:
            gb = remedy.size_bytes / _GB
            if remedy.kind == "pull":
                option_list.add_option(
                    f"install {remedy.model} on {remedy.provider} — {gb:.1f} GB · {remedy.note}"
                )
            else:
                option_list.add_option(f"(manual) {remedy.model} — {remedy.note}")

    # --- installation ---------------------------------------------------

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "remedies":
            return
        remedy = self._remedies[event.option_index]
        if remedy.kind != "pull":
            return  # hints are not actionable: there is no API for it
        status = self._discovery.status(remedy.provider) if self._discovery else None
        if status is None or not status.url:
            return

        self._cancel_pull = False
        gb = remedy.size_bytes / _GB
        # The download happens on the machine hosting Ollama, not on this
        # one: saying so before it starts is part of the confirmation.
        self.query_one("#pull_status", Static).update(
            f"Downloading {remedy.model} on {status.url} (~{gb:.1f} GB). x to cancel."
        )
        url, model = status.url, remedy.model
        self.run_worker(lambda: self._pull(url, model), thread=True, exclusive=True)

    def _pull(self, url: str, model: str) -> None:
        def on_progress(progress: PullProgress) -> None:
            if progress.total:
                text = (
                    f"{progress.status} "
                    f"{progress.completed / _GB:.1f}/{progress.total / _GB:.1f} GB"
                )
            else:
                text = progress.status
            self.app.call_from_thread(self._set_pull_status, text)

        error = pull_model(
            base_url=url,
            model=model,
            on_progress=on_progress,
            should_cancel=lambda: self._cancel_pull,
        )
        if error:
            self.app.call_from_thread(self._set_pull_status, f"✗ {error}")
            return
        self.app.call_from_thread(self._set_pull_status, "✓ installed")
        self.app.call_from_thread(self.action_refresh)
        if self._on_refresh is not None:
            self.app.call_from_thread(self._on_refresh)

    def _set_pull_status(self, text: str) -> None:
        self.query_one("#pull_status", Static).update(text)

    def action_cancel_pull(self) -> None:
        self._cancel_pull = True

    def action_close(self) -> None:
        self.dismiss(None)


class _DoctorApp(App):
    """Minimal app for the CLI subcommand: the doctor screen and nothing else."""

    def __init__(self, settings) -> None:
        super().__init__()
        self._settings = settings
        self.report: Optional[DoctorReport] = None

    def on_mount(self) -> None:
        self.push_screen(
            DoctorScreen(settings=self._settings, on_report=self._remember),
            callback=lambda _: self.exit(),
        )

    def _remember(self, report: DoctorReport) -> None:
        self.report = report


def run_doctor_cli(settings) -> int:
    """Run the doctor on its own and return the report's exit code."""
    app = _DoctorApp(settings)
    app.run(mouse=False)
    return app.report.exit_code if app.report is not None else 1
