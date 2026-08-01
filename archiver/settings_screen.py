from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, OptionList, Static, TextArea

from .archive_picker_screen import ArchivePickerResult, ArchivePickerScreen
from .llm_router import DS4_PREFIX
from .taxonomy import (
    get_default_taxonomy_for_language,
    get_effective_language,
    parse_taxonomy_lines,
)


@dataclass(frozen=True)
class SettingsResult:
    output_language: str
    taxonomies: dict[str, tuple[str, ...]]
    facts_model: str
    classify_model: str
    vision_model: str
    vision_model_fallback: str
    filename_separator: str
    ocr_mode: str
    undated_folder_name: str
    archive_root: Path
    ds4_base_url: str
    ollama_base_url: str


class SettingsScreen(ModalScreen[SettingsResult]):
    CSS = """
    SettingsScreen { layout: vertical; }
    #intro { height: auto; color: $text-muted; }
    #provider { height: auto; border: round $accent; background: $panel; padding: 1 2; }
    #options { height: auto; border: round $accent; background: $panel; }
    #taxonomy_label { height: auto; padding: 1 0 0 0; }
    #taxonomy { height: 1fr; border: round $accent; }
    #errors { height: auto; color: $error; }
    #ds4_label { height: auto; padding: 1 0 0 0; }
    #ds4_url { height: 3; }
    #ollama_label { height: auto; padding: 1 0 0 0; }
    #ollama_url { height: 3; }
    """

    BINDINGS = [
        ("ctrl+s", "save", "Save"),
        ("escape", "cancel", "Cancel"),
        ("q", "cancel", "Cancel"),
        ("o", "focus_options", "Options"),
        ("t", "focus_taxonomy", "Taxonomy"),
        ("r", "reset_taxonomy", "Reset taxonomy"),
    ]

    def __init__(
        self,
        *,
        output_language: str,
        taxonomies: dict[str, tuple[str, ...]],
        facts_model: str,
        classify_model: str,
        vision_model: str,
        vision_model_fallback: str,
        filename_separator: str,
        ocr_mode: str,
        undated_folder_name: str,
        archive_root: Path,
        ds4_base_url: str,
        ollama_base_url: str,
        available_models: tuple[str, ...],
        provider_info: str,
    ) -> None:
        super().__init__()
        self._provider_info = provider_info.strip()
        self._ds4_base_url = (ds4_base_url or "").strip()
        self._ollama_base_url = (ollama_base_url or "").strip() or "http://localhost:11434"
        self._output_language = output_language if output_language in {"auto", "it", "en"} else "auto"
        self._taxonomies: dict[str, tuple[str, ...]] = dict(taxonomies) if taxonomies else {}
        self._facts_model = facts_model or "auto"
        self._classify_model = classify_model or "auto"
        self._vision_model = vision_model or "auto"
        self._vision_model_fallback = vision_model_fallback or "none"
        self._archive_root = archive_root
        self._available_models = available_models
        self._filename_separator = filename_separator if filename_separator in {"space", "underscore", "dash"} else "space"
        self._ocr_mode = ocr_mode if ocr_mode in {"fast", "balanced", "high"} else "balanced"
        self._undated_folder_name = undated_folder_name.strip() if (undated_folder_name or "").strip() else "undated"

        self._text_options = ("auto",) + tuple(self._filter_text_models(available_models))
        self._vision_options = ("auto",) + tuple(self._filter_vision_models(available_models))
        self._vision_fallback_options = ("none", "auto") + tuple(self._filter_vision_models(available_models))
        self._lang_options = ("auto", "it", "en")
        self._sep_options = ("space", "underscore", "dash")
        self._ocr_options = ("fast", "balanced", "high")

    def _get_effective_lang(self) -> str:
        """Get the effective language for taxonomy display."""
        return get_effective_language(self._output_language)

    def _get_current_taxonomy_lines(self) -> tuple[str, ...]:
        """Get taxonomy lines for the current effective language."""
        lang = self._get_effective_lang()
        if lang in self._taxonomies and self._taxonomies[lang]:
            return self._taxonomies[lang]
        return get_default_taxonomy_for_language(lang)

    def _save_textarea_to_current_lang(self) -> None:
        """Save the current textarea content to the current language's taxonomy."""
        try:
            text = self.query_one("#taxonomy", TextArea).text
            lines = tuple(ln.rstrip("\n") for ln in text.splitlines() if ln.strip())
            lang = self._get_effective_lang()
            self._taxonomies[lang] = lines
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "Settings: ↑/↓ select • Enter/←/→ change • t taxonomy • Ctrl+S save • Esc cancel",
            id="intro",
        )
        yield Static(self._provider_info or "Provider: (unknown)", id="provider", markup=False)
        yield Static("ds4 endpoint (OpenAI-compatible, empty = disabled):", id="ds4_label")
        yield Input(value=self._ds4_base_url, placeholder="http://localhost:8000", id="ds4_url")
        yield Static("ollama endpoint:", id="ollama_label")
        yield Input(value=self._ollama_base_url, placeholder="http://localhost:11434", id="ollama_url")
        yield OptionList(*self._render_options(), id="options")
        lang = self._get_effective_lang()
        yield Static(f"Taxonomy [{lang.upper()}] (one category per line): name | description | examples", id="taxonomy_label")
        yield TextArea("\n".join(self._get_current_taxonomy_lines()).strip() + "\n", id="taxonomy", tab_behavior="indent")
        yield Static("", id="errors")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#options", OptionList).focus()

    def action_focus_options(self) -> None:
        self.query_one("#options", OptionList).focus()

    def action_focus_taxonomy(self) -> None:
        self.query_one("#taxonomy", TextArea).focus()

    def action_reset_taxonomy(self) -> None:
        lang = self._get_effective_lang()
        default_lines = get_default_taxonomy_for_language(lang)
        self.query_one("#taxonomy", TextArea).text = "\n".join(default_lines).strip() + "\n"
        self.query_one("#errors", Static).update("")

    def _current_ds4_url(self) -> str:
        try:
            return self.query_one("#ds4_url", Input).value.strip()
        except Exception:
            return self._ds4_base_url

    def _current_ollama_url(self) -> str:
        try:
            value = self.query_one("#ollama_url", Input).value.strip()
        except Exception:
            return self._ollama_base_url
        return value or "http://localhost:11434"

    def action_cancel(self) -> None:
        self.dismiss(
            SettingsResult(
                output_language=self._output_language,
                taxonomies=self._taxonomies,
                facts_model=self._facts_model,
                classify_model=self._classify_model,
                vision_model=self._vision_model,
                vision_model_fallback=self._vision_model_fallback,
                filename_separator=self._filename_separator,
                ocr_mode=self._ocr_mode,
                undated_folder_name=self._undated_folder_name,
                archive_root=self._archive_root,
                ds4_base_url=self._current_ds4_url(),
                ollama_base_url=self._current_ollama_url(),
            )
        )

    def action_save(self) -> None:
        self._save()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._activate_option(event.option_index)

    def on_key(self, event) -> None:  # type: ignore[override]
        if self.focused and self.focused.id == "options":
            if event.key in {"enter"}:
                ol = self.query_one("#options", OptionList)
                self._activate_option(ol.highlighted)
                event.stop()
            elif event.key in {"left", "right"}:
                ol = self.query_one("#options", OptionList)
                self._cycle_option(ol.highlighted, forward=(event.key == "right"))
                event.stop()
        elif self.focused and self.focused.id == "taxonomy":
            if event.key in {"escape"}:
                self.action_focus_options()
                event.stop()

    def _activate_option(self, idx: int) -> None:
        if idx in {0, 1, 2, 3, 4, 6, 7}:
            self._cycle_option(idx, forward=True)
            return
        if idx == 5:
            self.app.push_screen(
                ArchivePickerScreen(archive_root=self._archive_root),
                callback=self._on_archive_picked,
                wait_for_dismiss=False,
            )
            return
        if idx == 8:
            self._cycle_undated_name()
            return
        if idx == 9:
            self.action_focus_taxonomy()
            return

    def _cycle_option(self, idx: int, *, forward: bool) -> None:
        if idx == 0:
            self._facts_model = self._cycle_value(self._facts_model, self._text_options, forward=forward)
        elif idx == 1:
            self._classify_model = self._cycle_value(self._classify_model, self._text_options, forward=forward)
        elif idx == 2:
            self._vision_model = self._cycle_value(self._vision_model, self._vision_options, forward=forward)
        elif idx == 3:
            self._vision_model_fallback = self._cycle_value(self._vision_model_fallback, self._vision_fallback_options, forward=forward)
        elif idx == 4:
            # Language change: save current taxonomy, switch, load new
            self._save_textarea_to_current_lang()
            self._output_language = self._cycle_value(self._output_language, self._lang_options, forward=forward)
            self._update_taxonomy_display()
        elif idx == 6:
            self._filename_separator = self._cycle_value(self._filename_separator, self._sep_options, forward=forward)
        elif idx == 7:
            self._ocr_mode = self._cycle_value(self._ocr_mode, self._ocr_options, forward=forward)
        else:
            return
        self._refresh_options()

    def _update_taxonomy_display(self) -> None:
        """Update the taxonomy textarea and label for the current language."""
        lang = self._get_effective_lang()
        lines = self._get_current_taxonomy_lines()
        self.query_one("#taxonomy", TextArea).text = "\n".join(lines).strip() + "\n"
        self.query_one("#taxonomy_label", Static).update(
            f"Taxonomy [{lang.upper()}] (one category per line): name | description | examples"
        )
        self.query_one("#errors", Static).update("")

    def _cycle_undated_name(self) -> None:
        # Keep this simple for now: toggle between a few common choices.
        options = ("undated", "unknown-year", "senza-anno")
        cur = self._undated_folder_name
        try:
            i = options.index(cur)
            self._undated_folder_name = options[(i + 1) % len(options)]
        except Exception:
            self._undated_folder_name = options[0]
        self._refresh_options()

    def _on_archive_picked(self, result: ArchivePickerResult) -> None:
        self._archive_root = result.archive_root
        self._refresh_options()

    def _refresh_options(self) -> None:
        ol = self.query_one("#options", OptionList)
        highlighted = ol.highlighted
        ol.clear_options()
        ol.add_options(self._render_options())
        if highlighted is not None:
            ol.highlighted = highlighted

    def _render_options(self) -> list[str]:
        return [
            f"Facts model: {self._facts_model}",
            f"Classify model: {self._classify_model}",
            f"Vision model: {self._vision_model}",
            f"Vision fallback: {self._vision_model_fallback}",
            f"Output language: {self._output_language}",
            f"Archive folder: {self._archive_root}",
            f"Filename separator: {self._filename_separator}",
            f"OCR mode: {self._ocr_mode}",
            f"Undated folder: {self._undated_folder_name}",
            "Edit taxonomy (press Enter)",
        ]

    def _save(self) -> None:
        text = self.query_one("#taxonomy", TextArea).text
        lines = tuple(ln.rstrip("\n") for ln in text.splitlines() if ln.strip())
        _, errors = parse_taxonomy_lines(lines)
        if errors:
            self.query_one("#errors", Static).update("\n".join(errors[:6]))
            return
        # Save current textarea content to current language
        lang = self._get_effective_lang()
        self._taxonomies[lang] = lines
        self.dismiss(
            SettingsResult(
                output_language=self._output_language,
                taxonomies=self._taxonomies,
                facts_model=self._facts_model,
                classify_model=self._classify_model,
                vision_model=self._vision_model,
                vision_model_fallback=self._vision_model_fallback,
                filename_separator=self._filename_separator,
                ocr_mode=self._ocr_mode,
                undated_folder_name=self._undated_folder_name,
                archive_root=self._archive_root,
                ds4_base_url=self._current_ds4_url(),
                ollama_base_url=self._current_ollama_url(),
            )
        )

    @staticmethod
    def _cycle_value(current: str, values: tuple[str, ...], *, forward: bool) -> str:
        if current not in values:
            return values[0]
        i = values.index(current)
        if forward:
            return values[(i + 1) % len(values)]
        return values[(i - 1) % len(values)]

    @staticmethod
    def _filter_vision_models(models: tuple[str, ...]) -> list[str]:
        out: list[str] = []
        for m in models:
            ml = m.lower()
            if ml.startswith(DS4_PREFIX):
                continue
            if any(v in ml for v in ("vision", "llava", "moondream", "minicpm", "bakllava")):
                out.append(m)
        return out

    @staticmethod
    def _filter_text_models(models: tuple[str, ...]) -> list[str]:
        out: list[str] = []
        for m in models:
            ml = m.lower()
            if any(v in ml for v in ("vision", "llava", "moondream", "minicpm", "bakllava")):
                continue
            out.append(m)
        return out
