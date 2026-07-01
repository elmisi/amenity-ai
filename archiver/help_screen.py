from __future__ import annotations

from importlib import metadata

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static


class HelpScreen(ModalScreen[None]):
    CSS = """
    HelpScreen { layout: vertical; }
    #dialog { height: 1fr; border: round $accent; background: $panel; padding: 1 2; margin: 1 2; }
    #body { height: 1fr; }
    #hint { height: auto; color: $text-muted; }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("enter", "close", "Close"),
        ("q", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        try:
            ver = metadata.version("amenity-ai")
        except Exception:
            ver = "dev"

        yield Header(show_clock=False)
        with Container(id="dialog"):
            with VerticalScroll(id="body"):
                yield Static(
                    "\n".join(
                        [
                            f"amenity-ai v{ver}",
                            "Author: elmisi",
                            "",
                            "Terminal UI to organize files using a local LLM (via Ollama) with a 2-phase workflow:",
                            "",
                            "- extract high-signal facts (no classification yet),",
                            "- batch classify + propose coherent file names (taxonomy-driven).",
                            "",
                            "Upcoming milestones include per-file approval and applying rename/move operations",
                            "into an archive structured as {category}/{year}.",
                            "",
                            "Workflow:",
                            "1) Reload: load files from the source folder.",
                            "2) Scan: extract text/OCR + call the local LLM to produce structured facts and a rich summary.",
                            "3) Classify: use scanned facts to assign category/year and propose a coherent filename.",
                            "4) Archive: move classified (and skipped/error) files into the archive folder structure.",
                            "5) Review: inspect per-file details in the bottom panel.",
                            "",
                            "Shortcuts:",
                            "F1      Show this help screen",
                            "F2      Open settings (models, language, taxonomy, archive folder)",
                            "Ctrl+R  Reload directory and refresh the file list",
                            "Enter   Open selected file with the system default application",
                            "",
                            "s       Scan the selected row (force re-scan)",
                            "S       Scan all pending files",
                            "x       Stop an ongoing scan/classify task",
                            "",
                        "c       Classify the selected scanned row (force re-classify)",
                        "C       Classify all scanned files (per-file)",
                        "",
                        "u       Unclassify the selected row (keep scan results)",
                        "U       Unclassify all classified files (keeps scan results)",
                        "",
                        "m       Move the selected eligible file to the archive",
                        "M       Move all eligible files to the archive",
                            "",
                            "r       Reset the selected row back to pending (clears cached result)",
                            "R       Reset all rows back to pending (clears cache)",
                            "q       Quit",
                        ]
                    ),
                    markup=False,
                )
            yield Static("Enter / Esc / q to close", id="hint")
        yield Footer()

    def action_close(self) -> None:
        self.dismiss(None)
