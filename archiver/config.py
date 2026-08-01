from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    last_archive_root: Optional[str] = None
    last_source_root: Optional[str] = None
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

    def __post_init__(self) -> None:
        # Ensure taxonomies is always a dict
        if self.taxonomies is None:
            object.__setattr__(self, "taxonomies", {})


def _config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "amenity-stuff" / "config.json"


def _parse_taxonomy_list(lst: list) -> tuple[str, ...]:
    """Parse a list of taxonomy lines from JSON."""
    lines: list[str] = []
    for v in lst:
        if isinstance(v, str) and v.strip():
            lines.append(v.rstrip("\n"))
    return tuple(lines)


def load_config() -> AppConfig:
    from .taxonomy import get_effective_language

    path = _config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AppConfig()
    if not isinstance(data, dict):
        return AppConfig()
    last_archive = data.get("last_archive_root")
    last_source = data.get("last_source_root")
    output_language = data.get("output_language")
    taxonomies_raw = data.get("taxonomies")
    taxonomy_lines_legacy = data.get("taxonomy_lines")  # Old format
    facts_model = data.get("facts_model")
    classify_model = data.get("classify_model")
    legacy_text_model = data.get("text_model")
    vision_model = data.get("vision_model")
    vision_model_fallback = data.get("vision_model_fallback")
    filename_separator = data.get("filename_separator")
    ocr_mode = data.get("ocr_mode")
    undated_folder_name = data.get("undated_folder_name")
    ds4_base_url = data.get("ds4_base_url")

    kwargs: dict[str, object] = {}
    if isinstance(last_archive, str) and last_archive.strip():
        kwargs["last_archive_root"] = last_archive.strip()
    if isinstance(last_source, str) and last_source.strip():
        kwargs["last_source_root"] = last_source.strip()

    # Parse output_language first (needed for taxonomy migration)
    effective_lang = "en"
    if isinstance(output_language, str) and output_language.strip():
        lang = output_language.strip().lower()
        if lang in {"auto", "it", "en"}:
            kwargs["output_language"] = lang
            effective_lang = get_effective_language(lang)

    # Parse taxonomies (new format) or migrate from taxonomy_lines (old format)
    taxonomies: dict[str, tuple[str, ...]] = {}
    if isinstance(taxonomies_raw, dict):
        for lang_key, lines_list in taxonomies_raw.items():
            if isinstance(lang_key, str) and lang_key in {"it", "en"} and isinstance(lines_list, list):
                taxonomies[lang_key] = _parse_taxonomy_list(lines_list)
    elif isinstance(taxonomy_lines_legacy, list):
        # Migrate old format: assign to effective language
        lines = _parse_taxonomy_list(taxonomy_lines_legacy)
        if lines:
            taxonomies[effective_lang] = lines
    if taxonomies:
        kwargs["taxonomies"] = taxonomies

    # Backward compat: older configs used a single "text_model" for both phases.
    if isinstance(legacy_text_model, str) and legacy_text_model.strip():
        legacy = legacy_text_model.strip()
        kwargs["facts_model"] = legacy
        kwargs["classify_model"] = legacy
    if isinstance(facts_model, str) and facts_model.strip():
        kwargs["facts_model"] = facts_model.strip()
    if isinstance(classify_model, str) and classify_model.strip():
        kwargs["classify_model"] = classify_model.strip()
    if isinstance(vision_model, str) and vision_model.strip():
        kwargs["vision_model"] = vision_model.strip()
    if isinstance(vision_model_fallback, str) and vision_model_fallback.strip():
        kwargs["vision_model_fallback"] = vision_model_fallback.strip()
    if isinstance(filename_separator, str) and filename_separator.strip():
        sep = filename_separator.strip().lower()
        if sep in {"space", "underscore", "dash"}:
            kwargs["filename_separator"] = sep
    if isinstance(ocr_mode, str) and ocr_mode.strip():
        mode = ocr_mode.strip().lower()
        if mode in {"fast", "balanced", "high"}:
            kwargs["ocr_mode"] = mode
    if isinstance(undated_folder_name, str):
        val = undated_folder_name.strip()
        if val:
            kwargs["undated_folder_name"] = val
    if isinstance(ds4_base_url, str) and ds4_base_url.strip():
        kwargs["ds4_base_url"] = ds4_base_url.strip()
    return AppConfig(**kwargs)  # type: ignore[arg-type]


def save_config(config: AppConfig) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert taxonomies tuples to lists for JSON serialization
    data = dict(config.__dict__)
    if "taxonomies" in data and isinstance(data["taxonomies"], dict):
        data["taxonomies"] = {k: list(v) for k, v in data["taxonomies"].items()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
