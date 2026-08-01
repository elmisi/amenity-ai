from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    source_root: Path
    archive_root: Path
    recursive: bool = True
    include_extensions: tuple[str, ...] = (
        "pdf",
        "jpg",
        "jpeg",
        "png",
        "doc",
        "docx",
        "odt",
        "xls",
        "xlsx",
        "json",
        "md",
        "txt",
        "rtf",
        "svg",
        "kmz",
        "gpx",
        "html",
        "htm",
        "csv",
        "yaml",
        "yml",
    )
    exclude_dirnames: tuple[str, ...] = (".git", ".venv", ".amenity-stuff", "ARCHIVE")
    output_language: str = "auto"  # auto | it | en
    taxonomies: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]  # {lang: lines}
    facts_model: str = "auto"
    classify_model: str = "auto"
    vision_model: str = "auto"
    vision_model_fallback: str = "none"  # none | auto | llava:7b | minicpm-v | ...
    filename_separator: str = "space"  # space | underscore | dash
    ocr_mode: str = "balanced"  # fast | balanced | high
    undated_folder_name: str = "undated"
    ds4_base_url: str = ""  # OpenAI-compatible endpoint; empty = disabled
    ollama_base_url: str = "http://localhost:11434"  # may point at another machine
    skip_initial_setup: bool = False

    def __post_init__(self) -> None:
        # Ensure taxonomies is always a dict
        if self.taxonomies is None:
            object.__setattr__(self, "taxonomies", {})

    def get_taxonomy_lines(self) -> tuple[str, ...]:
        """Get the taxonomy lines for the effective language."""
        from .taxonomy import get_default_taxonomy_for_language, get_effective_language
        effective_lang = get_effective_language(self.output_language)
        if effective_lang in self.taxonomies and self.taxonomies[effective_lang]:
            return self.taxonomies[effective_lang]
        return get_default_taxonomy_for_language(effective_lang)
