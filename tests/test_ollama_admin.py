from __future__ import annotations

import io
import json

from archiver.ollama_admin import PullProgress, probe_vision, pull_model


def _stream(lines):
    payload = b"".join((json.dumps(line) + "\n").encode("utf-8") for line in lines)

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(request, timeout=None):
        opener.request = request
        return _Resp(payload)

    return opener


def test_fraction_is_zero_when_total_is_unknown():
    assert PullProgress(status="pulling", completed=5, total=0).fraction == 0.0
    assert PullProgress(status="pulling", completed=5, total=10).fraction == 0.5


def test_successful_pull_reports_progress_and_returns_none():
    seen = []
    opener = _stream([
        {"status": "pulling manifest"},
        {"status": "pulling", "completed": 500, "total": 1000},
        {"status": "success"},
    ])

    error = pull_model(base_url="http://ollama.invalid", model="llava:7b",
                       on_progress=seen.append, opener=opener)

    assert error is None
    assert seen[-1].status == "success"
    assert any(p.fraction == 0.5 for p in seen)


def test_pull_posts_the_model_name_to_the_api_pull_endpoint():
    opener = _stream([{"status": "success"}])
    pull_model(base_url="http://ollama.invalid/", model="llava:7b", opener=opener)
    assert opener.request.full_url == "http://ollama.invalid/api/pull"
    assert json.loads(opener.request.data.decode("utf-8"))["model"] == "llava:7b"


def test_error_line_in_the_stream_is_returned():
    opener = _stream([{"error": "model not found"}])
    assert pull_model(base_url="http://ollama.invalid", model="nope", opener=opener) == "model not found"


def test_cancellation_stops_the_stream_and_reports_it():
    opener = _stream([
        {"status": "pulling", "completed": 1, "total": 100},
        {"status": "pulling", "completed": 2, "total": 100},
        {"status": "success"},
    ])
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 1

    error = pull_model(base_url="http://ollama.invalid", model="m",
                       should_cancel=should_cancel, opener=opener)
    assert error == "cancelled"


def test_transport_failure_is_returned_as_a_message():
    def opener(request, timeout=None):
        raise ConnectionRefusedError("nope")

    assert "ConnectionRefusedError" in pull_model(
        base_url="http://ollama.invalid", model="m", opener=opener
    )


def test_probe_vision_maps_status_codes_through_interpret_probe():
    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(request, timeout=None):
        return _Resp(b'{"choices":[{"message":{"content":"ok"}}]}')

    assert probe_vision(base_url="http://vllm.invalid", model="m", opener=opener) is True
