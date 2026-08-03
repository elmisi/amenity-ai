from contextlib import contextmanager

from archiver import llm_router
from archiver.llm_backend import LLMResponse


class RecordingLimiter:
    """Stands in for ConcurrencyLimiter and records the order of events."""

    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def slot(self, provider: str):
        self.events.append(f"enter:{provider}")
        try:
            yield
        finally:
            self.events.append(f"exit:{provider}")


def _fake_backend(events):
    class Backend:
        def generate(self, **kwargs):
            events.append("call")
            return LLMResponse(text="{}", model="m", done=True, error=None)

    return Backend()


def test_the_slot_wraps_the_call_and_names_the_resolved_provider(monkeypatch):
    limiter = RecordingLimiter()
    monkeypatch.setattr(
        llm_router,
        "_resolve",
        lambda model, urls: (
            _fake_backend(limiter.events),
            llm_router.split_model_id(model)[0],
            "bare",
        ),
    )
    llm_router.generate(
        model="vllm:whatever",
        prompt="p",
        provider_urls={"vllm": "http://example.invalid"},
        limiter=limiter,
    )
    assert limiter.events == ["enter:vllm", "call", "exit:vllm"]


def test_the_slot_is_released_when_the_backend_raises(monkeypatch):
    limiter = RecordingLimiter()

    class Boom:
        def generate(self, **kwargs):
            raise RuntimeError("network died")

    monkeypatch.setattr(
        llm_router,
        "_resolve",
        lambda model, urls: (Boom(), llm_router.split_model_id(model)[0], "bare"),
    )
    try:
        llm_router.generate(
            model="vllm:whatever",
            prompt="p",
            provider_urls={"vllm": "http://example.invalid"},
            limiter=limiter,
        )
    except RuntimeError:
        pass
    assert limiter.events == ["enter:vllm", "exit:vllm"]


def test_without_a_limiter_nothing_changes(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        llm_router,
        "_resolve",
        lambda model, urls: (
            _fake_backend(events),
            llm_router.split_model_id(model)[0],
            "bare",
        ),
    )
    result = llm_router.generate(
        model="vllm:whatever", prompt="p", provider_urls={"vllm": "http://example.invalid"}
    )
    assert events == ["call"]
    assert result.error is None


def test_an_unconfigured_provider_never_takes_a_slot():
    limiter = RecordingLimiter()
    result = llm_router.generate(
        model="ds4:model", prompt="p", provider_urls={"ds4": ""}, limiter=limiter
    )
    assert result.error == "ds4: endpoint not configured"
    assert limiter.events == []


def test_the_image_path_forwards_the_limiter(monkeypatch, tmp_path):
    limiter = RecordingLimiter()
    monkeypatch.setattr(
        llm_router,
        "_resolve",
        lambda model, urls: (
            _fake_backend(limiter.events),
            llm_router.split_model_id(model)[0],
            "bare",
        ),
    )
    image = tmp_path / "pic.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    llm_router.generate_with_image_file(
        model="vllm:whatever",
        prompt="describe",
        image_path=str(image),
        provider_urls={"vllm": "http://example.invalid"},
        limiter=limiter,
    )
    assert limiter.events == ["enter:vllm", "call", "exit:vllm"]
