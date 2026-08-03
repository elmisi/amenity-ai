from pathlib import Path

from archiver.capabilities import CAP_COMPLETION, CAP_VISION, SOURCE_DECLARED
from archiver.discovery import DiscoveryResult, ModelInfo, ProviderStatus
from archiver.settings import Settings
from archiver.ui_runtime import banner_for_state, runtime_problem


def _settings() -> Settings:
    return Settings(source_root=Path("/tmp/s"), archive_root=Path("/tmp/a"))


def _model(mid: str, *, vision: bool) -> ModelInfo:
    caps = {CAP_COMPLETION} | ({CAP_VISION} if vision else set())
    return ModelInfo(
        id=mid,
        provider=mid.split(":")[0],
        capabilities=frozenset(caps),
        capability_source=SOURCE_DECLARED,
        parameter_size_b=27.0,
    )


def _discovery(*statuses: ProviderStatus) -> DiscoveryResult:
    return DiscoveryResult(providers=tuple(statuses))


def _up(name: str, *models: ModelInfo) -> ProviderStatus:
    return ProviderStatus(
        name=name, url="http://example.invalid", configured=True, available=True,
        models=tuple(models),
    )


def test_no_discovery_yet_is_informational():
    assert runtime_problem(None, _settings()) == ("Detecting providers…", "info")


def test_a_covered_setup_is_clean():
    ok = _up("vllm", _model("vllm:m", vision=True))
    assert runtime_problem(_discovery(ok), _settings()) == (None, "ok")


def test_a_provider_that_is_down_only_warns_when_the_roles_are_covered():
    """The whole point of the change: Ollama stopped on purpose must not paint
    the banner red while vLLM covers both roles."""
    down = ProviderStatus(
        name="ollama", url="http://example.invalid", configured=True, available=False,
        detail="URLError", models=(),
    )
    message, severity = runtime_problem(
        _discovery(_up("vllm", _model("vllm:m", vision=True)), down), _settings()
    )
    assert severity == "warn"
    assert "ollama" in message


def test_no_semantic_model_is_an_error():
    message, severity = runtime_problem(_discovery(_up("vllm")), _settings())
    assert severity == "error"
    assert message == "No semantic model available"


def test_a_text_only_setup_warns_about_vision():
    ok = _up("vllm", _model("vllm:m", vision=False))
    message, severity = runtime_problem(_discovery(ok), _settings())
    assert severity == "warn"
    assert "vision" in message


def test_an_unconfigured_provider_is_not_a_problem():
    skipped = ProviderStatus(name="ds4", url="", configured=False)
    ok = _up("vllm", _model("vllm:m", vision=True))
    assert runtime_problem(_discovery(ok, skipped), _settings()) == (None, "ok")


def test_a_warning_does_not_replace_the_running_banner():
    text, _ = banner_for_state(
        state="scanning…", scanning=4, classifying=0, moving=0,
        problem="ollama unreachable", severity="warn",
    )
    assert "RUNNING" in text
    assert "ollama unreachable" in text


def test_the_running_banner_says_how_many_are_in_flight():
    text, _ = banner_for_state(
        state="scanning…", scanning=4, classifying=0, moving=0, problem=None, severity="ok",
    )
    assert "4 in flight" in text


def test_stopping_says_how_many_it_is_waiting_for():
    text, _ = banner_for_state(
        state="stopping…", scanning=3, classifying=0, moving=0, problem=None, severity="ok",
    )
    assert text == "STOPPING — waiting for 3 requests in flight"


def test_stopping_with_nothing_left_says_so_plainly():
    text, _ = banner_for_state(
        state="stopping…", scanning=0, classifying=0, moving=0, problem=None, severity="ok",
    )
    assert text == "STOPPING…"
