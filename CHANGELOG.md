# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.9.13] - 2026-07-01

### Changed
- Renamed the project from `amenity-stuff` to `amenity-ai`. This affects the GitHub
  repository, the Python package name, and the CLI command (`amenity-stuff` → `amenity-ai`).
- Updated documentation, install/uninstall scripts, and taxonomy file headers to the new name.

### Notes
- Runtime directories are intentionally **unchanged** for backward compatibility, so existing
  settings and caches keep working:
  - user config: `~/.config/amenity-stuff/`
  - per-source / per-archive caches: `.amenity-stuff/`
  - install location: `~/.local/share/amenity-stuff/`
- After upgrading, reinstall to pick up the new command name:
  `~/.local/share/amenity-stuff/venv/bin/pip install -e .`
