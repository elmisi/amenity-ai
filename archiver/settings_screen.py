from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, OptionList, Static, TextArea

from .archive_picker_screen import ArchivePickerResult, ArchivePickerScreen
from .concurrency import clamp_limit
from .model_selection import ROLE_CLASSIFY, ROLE_FACTS, ROLE_VISION, rank_models
from .providers import PROVIDER_NAMES, PROVIDERS
from .taxonomy import (
    get_default_taxonomy_for_language,
    get_effective_language,
    parse_taxonomy_lines,
)

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import ModelInfo


@dataclass(frozen=True)
class SettingsResult:
    output_language: str
    taxonomies: dict[str, tuple[str, ...]]
    facts_model: str
    classify_model: str
    vision_model: str
    filename_separator: str
    ocr_mode: str
    undated_folder_name: str
    archive_root: Path
    providers: dict[str, str]
    provider_concurrency: dict[str, int]


class SettingsScreen(ModalScreen[SettingsResult]):
    CSS = """
    SettingsScreen { layout: vertical; }
    #intro { height: auto; color: $text-muted; }
    #provider { height: auto; background: $panel; padding: 0 1; }
    #options { height: auto; border: round $accent; background: $panel; }
    #taxonomy_label { height: auto; }
    #taxonomy { height: 1fr; border: round $accent; }
    #errors { height: auto; color: $error; }
    .provider_row { height: 1; }
    .provider_name { width: 8; color: $text-muted; padding: 0 1; }
    .provider_url { width: 1fr; height: 1; border: none; padding: 0 1; }
    .provider_par { width: 10; height: 1; border: none; padding: 0 1; }
    .provider_url:focus, .provider_par:focus { background: $accent 30%; }
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
        filename_separator: str,
        ocr_mode: str,
        undated_folder_name: str,
        archive_root: Path,
        providers: dict[str, str],
        provider_concurrency: dict[str, int],
        available_models: tuple["ModelInfo", ...],
        provider_info: str,
    ) -> None:
        super().__init__()
        self._provider_info = provider_info.strip()
        self._providers = {name: (providers or {}).get(name, "") for name in PROVIDER_NAMES}
        self._concurrency = {
            spec.name: clamp_limit(
                (provider_concurrency or {}).get(spec.name), default=spec.max_concurrency
            )
            for spec in PROVIDERS
        }
        self._output_language = output_language if output_language in {"auto", "it", "en"} else "auto"
        self._taxonomies: dict[str, tuple[str, ...]] = dict(taxonomies) if taxonomies else {}
        self._facts_model = facts_model or "auto"
        self._classify_model = classify_model or "auto"
        self._vision_model = vision_model or "auto"
        self._archive_root = archive_root
        self._available_models = available_models
        self._filename_separator = filename_separator if filename_separator in {"space", "underscore", "dash"} else "space"
        self._ocr_mode = ocr_mode if ocr_mode in {"fast", "balanced", "high"} else "balanced"
        self._undated_folder_name = undated_folder_name.strip() if (undated_folder_name or "").strip() else "undated"

        self._facts_options = ("auto",) + rank_models(available_models, ROLE_FACTS)
        self._classify_options = ("auto",) + rank_models(available_models, ROLE_CLASSIFY)
        self._vision_options = ("auto",) + rank_models(available_models, ROLE_VISION)
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
        # One line per provider: name, URL, parallel slots. The default URL
        # lives in the placeholder, so no label line is needed above.
        for spec in PROVIDERS:
            placeholder = spec.default_url or "empty = disabled"
            with Horizontal(classes="provider_row"):
                yield Static(spec.name, classes="provider_name")
                yield Input(value=self._providers.get(spec.name, ""),
                            placeholder=placeholder,
                            classes="provider_url", id=f"{spec.name}_url")
                yield Input(value=str(self._concurrency[spec.name]),
                            placeholder="par",
                            classes="provider_par", id=f"{spec.name}_par")
        yield OptionList(*self._render_options(), id="options")
        lang = self._get_effective_lang()
        # markup=False: Rich would otherwise eat "[EN]" as a style tag.
        yield Static(f"Taxonomy [{lang.upper()}] (one category per line): name | description | examples", id="taxonomy_label", markup=False)
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

    def _current_provider_urls(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in PROVIDERS:
            try:
                value = self.query_one(f"#{spec.name}_url", Input).value.strip()
            except Exception:
                value = self._providers.get(spec.name, "")
            out[spec.name] = value or spec.default_url
        return out

    def _current_provider_concurrency(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for spec in PROVIDERS:
            try:
                raw = self.query_one(f"#{spec.name}_par", Input).value.strip()
            except Exception:
                raw = ""
            out[spec.name] = clamp_limit(raw, default=self._concurrency[spec.name])
        return out

    def action_cancel(self) -> None:
        self.dismiss(
            SettingsResult(
                output_language=self._output_language,
                taxonomies=self._taxonomies,
                facts_model=self._facts_model,
                classify_model=self._classify_model,
                vision_model=self._vision_model,
                filename_separator=self._filename_separator,
                ocr_mode=self._ocr_mode,
                undated_folder_name=self._undated_folder_name,
                archive_root=self._archive_root,
                providers=self._current_provider_urls(),
                provider_concurrency=self._current_provider_concurrency(),
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
        if idx in {0, 1, 2, 3, 5, 6}:
            self._cycle_option(idx, forward=True)
            return
        if idx == 4:
            self.app.push_screen(
                ArchivePickerScreen(archive_root=self._archive_root),
                callback=self._on_archive_picked,
                wait_for_dismiss=False,
            )
            return
        if idx == 7:
            self._cycle_undated_name()
            return
        if idx == 8:
            self.action_focus_taxonomy()
            return

    def _cycle_option(self, idx: int, *, forward: bool) -> None:
        if idx == 0:
            self._facts_model = self._cycle_value(self._facts_model, self._facts_options, forward=forward)
        elif idx == 1:
            self._classify_model = self._cycle_value(self._classify_model, self._classify_options, forward=forward)
        elif idx == 2:
            self._vision_model = self._cycle_value(self._vision_model, self._vision_options, forward=forward)
        elif idx == 3:
            # Language change: save current taxonomy, switch, load new
            self._save_textarea_to_current_lang()
            self._output_language = self._cycle_value(self._output_language, self._lang_options, forward=forward)
            self._update_taxonomy_display()
        elif idx == 5:
            self._filename_separator = self._cycle_value(self._filename_separator, self._sep_options, forward=forward)
        elif idx == 6:
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
                filename_separator=self._filename_separator,
                ocr_mode=self._ocr_mode,
                undated_folder_name=self._undated_folder_name,
                archive_root=self._archive_root,
                providers=self._current_provider_urls(),
                provider_concurrency=self._current_provider_concurrency(),
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
