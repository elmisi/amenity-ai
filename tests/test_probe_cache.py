from __future__ import annotations

import json

from archiver.capabilities import CAP_COMPLETION, CAP_VISION
from archiver.probe_cache import (
    load_probe_cache,
    probe_cache_path,
    prune_probe_cache,
    save_probe_result,
)


def test_path_follows_xdg_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert probe_cache_path() == tmp_path / "amenity-stuff" / "probe_cache.json"


def test_missing_file_yields_empty_cache(tmp_path):
    assert load_probe_cache(tmp_path / "nope.json") == {}


def test_corrupt_file_yields_empty_cache_instead_of_raising(tmp_path):
    path = tmp_path / "probe_cache.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_probe_cache(path) == {}


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "probe_cache.json"
    save_probe_result(
        url="http://example.invalid:8000",
        bare_id="qwen3.6-27b",
        capabilities=frozenset({CAP_COMPLETION, CAP_VISION}),
        path=path,
    )
    cache = load_probe_cache(path)
    assert cache[("http://example.invalid:8000", "qwen3.6-27b")] == frozenset(
        {CAP_COMPLETION, CAP_VISION}
    )


def test_save_normalises_trailing_slash_in_url(tmp_path):
    path = tmp_path / "probe_cache.json"
    save_probe_result(
        url="http://example.invalid:8000/",
        bare_id="m",
        capabilities=frozenset({CAP_COMPLETION}),
        path=path,
    )
    assert ("http://example.invalid:8000", "m") in load_probe_cache(path)


def test_prune_drops_entries_whose_model_disappeared(tmp_path):
    path = tmp_path / "probe_cache.json"
    save_probe_result(url="http://a.invalid", bare_id="gone",
                      capabilities=frozenset({CAP_COMPLETION}), path=path)
    save_probe_result(url="http://a.invalid", bare_id="kept",
                      capabilities=frozenset({CAP_COMPLETION}), path=path)

    prune_probe_cache(known={("http://a.invalid", "kept")}, path=path)

    cache = load_probe_cache(path)
    assert set(cache) == {("http://a.invalid", "kept")}


def test_file_is_written_atomically_as_json_object(tmp_path):
    path = tmp_path / "probe_cache.json"
    save_probe_result(url="http://a.invalid", bare_id="m",
                      capabilities=frozenset({CAP_VISION}), path=path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert not list(tmp_path.glob("*.tmp"))
