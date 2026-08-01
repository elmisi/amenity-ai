from __future__ import annotations

from .config import AppConfig
from .settings import Settings
from .setup_screen import SetupResult


def settings_from_setup(*, current: Settings, setup: SetupResult) -> Settings:
    return Settings(
        source_root=setup.source_root,
        archive_root=setup.archive_root,
        recursive=current.recursive,
        include_extensions=current.include_extensions,
        exclude_dirnames=current.exclude_dirnames,
        output_language=current.output_language,
        taxonomies=current.taxonomies,
        facts_model=current.facts_model,
        classify_model=current.classify_model,
        vision_model=current.vision_model,
        filename_separator=current.filename_separator,
        ocr_mode=current.ocr_mode,
        undated_folder_name=current.undated_folder_name,
        providers=current.providers,
        skip_initial_setup=current.skip_initial_setup,
    )


def app_config_from_settings(settings: Settings) -> AppConfig:
    return AppConfig(
        last_archive_root=str(settings.archive_root),
        last_source_root=str(settings.source_root),
        output_language=settings.output_language,
        taxonomies=settings.taxonomies,
        facts_model=settings.facts_model,
        classify_model=settings.classify_model,
        vision_model=settings.vision_model,
        filename_separator=settings.filename_separator,
        ocr_mode=settings.ocr_mode,
        undated_folder_name=settings.undated_folder_name,
        providers=settings.providers,
    )

