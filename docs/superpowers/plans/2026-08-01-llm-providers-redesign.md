# Ristrutturazione provider LLM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sostituire l'attuale gestione a due provider LLM con un registry a tre (vLLM, Ollama, ds4) che scopre i modelli da sé a partire da tre sole URL, li ordina sui metadati reali invece che su liste hardcoded, e offre una modalità `doctor` che diagnostica i buchi e sa installare i modelli mancanti.

**Architecture:** Un registry puro (`providers.py`) è l'unica fonte di verità su nomi, prefissi, priorità e quirk dei provider. Sopra ci stanno tre livelli puri e testabili senza rete — capability (`capabilities.py`), ranking (`model_selection.py`), diagnosi (`doctor.py`) — e due livelli che fanno I/O (`discovery.py`, `ollama_admin.py`). La UI Textual (`doctor_screen.py`, `settings_screen.py`) è l'ultimo strato e non contiene logica.

**Tech Stack:** Python 3, `urllib.request` (nessuna nuova dipendenza), `concurrent.futures.ThreadPoolExecutor`, Textual, pytest (dev-only).

**Spec di riferimento:** `docs/superpowers/specs/2026-08-01-llm-providers-redesign-design.md`

**Branch:** `feat/llm-providers-redesign` (già creato, contiene il commit della spec)

## Global Constraints

Valgono per ogni task, implicitamente incluse nei requisiti di ognuna.

- **Nessun hostname reale nel repository.** I default sono `http://localhost:11434` per Ollama e stringa vuota per vLLM e ds4. Gli endpoint reali vivono solo nella config locale dell'utente. Questo vale anche per i commenti, i docstring e i messaggi di commit.
- **Nessuna nuova dipendenza runtime.** HTTP con `urllib.request`, come tutto il resto del progetto. `pyproject.toml` non cambia se non per la versione.
- **Dataclass frozen** per config, risultati e stato. Dove serve un default mutabile si usa l'idioma già presente nel progetto: campo dichiarato `= None` con `__post_init__` che normalizza via `object.__setattr__` (vedi `settings.py:50`).
- **`pathlib.Path`** per i percorsi su filesystem; type hints su tutte le funzioni pubbliche.
- **Mai bloccare l'event loop di Textual**: I/O, OCR e chiamate LLM girano in worker; gli aggiornamenti UI passano da `call_from_thread`.
- **Cancellazione cooperativa** via callback `should_cancel`, come già fanno scan, classify e move.
- **Prompt, euristiche di estrazione e flusso UX invariati.** Questa ristrutturazione tocca la scelta e l'instradamento dei modelli, non cosa viene chiesto loro.
- **Messaggi di commit** in forma `type: description` (es. `feat: add provider registry`). **Mai** `Co-Authored-By: Claude` né altre attribuzioni ad agenti.
- **Versione**: `VERSION` e `pyproject.toml` devono restare in sincrono. Il bump a `0.12.0` avviene **una sola volta, nell'ultima task**. Non bumpare a ogni commit.
- **Test**: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`. pytest è dev-only e non va aggiunto a `pyproject.toml`. Nessun test sulla TUI.
- **Suite verde a ogni commit.** I test esistenti che riguardano l'assetto a due provider vengono riscritti nella task che ne cambia il contratto, non lasciati rotti per dopo.

---

### Task 1: Registry dei provider

Modulo puro, senza I/O e senza dipendenze interne: è la base di tutto il resto.

**Files:**
- Create: `archiver/providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: niente
- Produces:
  - `ProviderSpec` (frozen dataclass): `name: str`, `kind: str`, `prefix: str`, `default_url: str = ""`, `sends_reasoning_effort: bool = False`, `supports_install: bool = False`
  - `PROVIDERS: tuple[ProviderSpec, ...]` — ordine di dichiarazione = priorità
  - `PROVIDER_NAMES: tuple[str, ...]`
  - `provider_by_name(name: str) -> ProviderSpec | None`
  - `provider_priority(name: str) -> int`
  - `split_model_id(model_id: str) -> tuple[ProviderSpec, str]`
  - `join_model_id(provider_name: str, bare_id: str) -> str`
  - `default_provider_urls() -> dict[str, str]`

- [ ] **Step 1: Write the failing test**

Crea `tests/test_providers.py`:

```python
from __future__ import annotations

import pytest

from archiver.providers import (
    PROVIDERS,
    PROVIDER_NAMES,
    default_provider_urls,
    join_model_id,
    provider_by_name,
    provider_priority,
    split_model_id,
)


def test_priority_order_is_vllm_then_ollama_then_ds4():
    assert [p.name for p in PROVIDERS] == ["vllm", "ollama", "ds4"]
    assert provider_priority("vllm") < provider_priority("ollama") < provider_priority("ds4")


def test_split_uses_known_prefix_not_first_colon():
    spec, bare = split_model_id("ollama:qwen3:8b")
    assert spec.name == "ollama"
    assert bare == "qwen3:8b"


def test_split_treats_bare_legacy_id_as_ollama():
    # Config 0.11.0 salvavano "qwen3:8b" senza prefisso: non deve mai
    # produrre un provider inesistente "qwen3".
    spec, bare = split_model_id("qwen3:8b")
    assert spec.name == "ollama"
    assert bare == "qwen3:8b"


def test_split_recognises_openai_compat_prefixes():
    assert split_model_id("vllm:qwen3.6-27b")[0].name == "vllm"
    assert split_model_id("vllm:qwen3.6-27b")[1] == "qwen3.6-27b"
    assert split_model_id("ds4:deepseek-v4-flash")[0].name == "ds4"


@pytest.mark.parametrize("provider,bare", [
    ("ollama", "qwen3:8b"),
    ("vllm", "qwen3.6-27b"),
    ("ds4", "deepseek-v4-flash"),
])
def test_join_split_round_trip(provider, bare):
    joined = join_model_id(provider, bare)
    spec, out = split_model_id(joined)
    assert (spec.name, out) == (provider, bare)


def test_only_ollama_supports_install_and_only_ds4_sends_reasoning_effort():
    assert [p.name for p in PROVIDERS if p.supports_install] == ["ollama"]
    assert [p.name for p in PROVIDERS if p.sends_reasoning_effort] == ["ds4"]


def test_default_urls_have_no_real_hostnames():
    urls = default_provider_urls()
    assert set(urls) == set(PROVIDER_NAMES)
    assert urls["ollama"] == "http://localhost:11434"
    assert urls["vllm"] == ""
    assert urls["ds4"] == ""


def test_provider_by_name_returns_none_for_unknown():
    assert provider_by_name("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_providers.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'archiver.providers'`

- [ ] **Step 3: Write minimal implementation**

Crea `archiver/providers.py`:

```python
"""Registry dei provider LLM.

Unica fonte di verità su nomi, prefissi, priorità e differenze di
comportamento fra provider. L'ordine di dichiarazione di PROVIDERS È la
priorità usata dal ranking: vllm regge la contesa e scalerà quando la
scansione verrà parallelizzata; ollama è sempre disponibile; ds4 è
mutuamente esclusivo, quindi una scansione lunga lo monopolizzerebbe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

KIND_OLLAMA = "ollama"
KIND_OPENAI_COMPAT = "openai_compat"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: str
    prefix: str
    default_url: str = ""
    sends_reasoning_effort: bool = False
    supports_install: bool = False


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec("vllm", KIND_OPENAI_COMPAT, "vllm:"),
    ProviderSpec(
        "ollama",
        KIND_OLLAMA,
        "ollama:",
        default_url="http://localhost:11434",
        supports_install=True,
    ),
    ProviderSpec("ds4", KIND_OPENAI_COMPAT, "ds4:", sends_reasoning_effort=True),
)

PROVIDER_NAMES: tuple[str, ...] = tuple(p.name for p in PROVIDERS)

# Gli id senza prefisso noto vengono da config o cache scritte prima della
# 0.12.0, quando "nudo" significava Ollama.
_LEGACY_SPEC = next(p for p in PROVIDERS if p.name == "ollama")


def provider_by_name(name: str) -> Optional[ProviderSpec]:
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None


def provider_priority(name: str) -> int:
    for index, spec in enumerate(PROVIDERS):
        if spec.name == name:
            return index
    return len(PROVIDERS)


def split_model_id(model_id: str) -> tuple[ProviderSpec, str]:
    """Separa prefisso e id nudo confrontando con i prefissi NOTI.

    Non spezza sul primo ':' incontrato: "ollama:qwen3:8b" deve dare
    ("ollama", "qwen3:8b") e "qwen3:8b" non deve dare un provider "qwen3".
    """
    for spec in PROVIDERS:
        if model_id.startswith(spec.prefix):
            return spec, model_id[len(spec.prefix):]
    return _LEGACY_SPEC, model_id


def join_model_id(provider_name: str, bare_id: str) -> str:
    spec = provider_by_name(provider_name)
    if spec is None:
        raise KeyError(f"unknown provider: {provider_name}")
    return spec.prefix + bare_id


def default_provider_urls() -> dict[str, str]:
    return {spec.name: spec.default_url for spec in PROVIDERS}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_providers.py -v`
Expected: PASS, 9 test

- [ ] **Step 5: Commit**

```bash
git add archiver/providers.py tests/test_providers.py
git commit -m "feat: add provider registry with explicit prefixes and priority"
```

---

### Task 2: Capability e parsing della taglia

Modulo puro. Nessuna rete: il probe qui è solo l'**interpretazione** di una risposta, la chiamata HTTP arriva nella Task 12.

**Files:**
- Create: `archiver/capabilities.py`
- Test: `tests/test_capabilities.py`

**Interfaces:**
- Consumes: niente
- Produces:
  - `CAP_COMPLETION = "completion"`, `CAP_VISION = "vision"`, `CAP_EMBEDDING = "embedding"`
  - `SOURCE_DECLARED = "declared"`, `SOURCE_HEURISTIC = "heuristic"`, `SOURCE_PROBED = "probed"`
  - `parse_parameter_size(*texts: str) -> float | None`
  - `guess_capabilities(*, model_id: str, root: str = "") -> frozenset[str]`
  - `interpret_probe(*, status: int, body: str) -> bool | None`

- [ ] **Step 1: Write the failing test**

Crea `tests/test_capabilities.py`:

```python
from __future__ import annotations

from archiver.capabilities import (
    CAP_COMPLETION,
    CAP_VISION,
    guess_capabilities,
    interpret_probe,
    parse_parameter_size,
)


def test_size_from_vllm_id_ignores_the_version_number():
    # "qwen3.6-27b": il 3.6 è la versione, il 27 è la taglia.
    assert parse_parameter_size("qwen3.6-27b") == 27.0


def test_size_from_vllm_root_path():
    assert parse_parameter_size("/models/Qwen3.6-27B-AWQ-INT4") == 27.0


def test_size_from_ollama_parameter_size_field():
    assert parse_parameter_size("8.2B") == 8.2


def test_size_from_ollama_tag():
    assert parse_parameter_size("gemma3:1b") == 1.0
    assert parse_parameter_size("llama3.3:70b") == 70.0


def test_size_is_none_when_nothing_says_it():
    assert parse_parameter_size("phi4-mini:latest") is None
    assert parse_parameter_size("") is None


def test_size_takes_first_match_across_arguments_in_order():
    # root ha la precedenza sull'id perché è più informativo
    assert parse_parameter_size("", "/models/Qwen-235B-A22B", "qwen") == 235.0


def test_vision_heuristic_recognises_known_families():
    for name in ("llava:7b", "moondream:latest", "minicpm-v:latest",
                 "bakllava:latest", "Qwen2.5-VL-7B-Instruct", "pixtral-12b"):
        assert CAP_VISION in guess_capabilities(model_id=name), name


def test_vision_heuristic_misses_qwen36_27b_which_is_why_the_probe_exists():
    # Falso negativo verificato sul campo il 2026-08-01: il modello ACCETTA
    # immagini ma il nome non lo dice. Solo il probe può correggerlo.
    caps = guess_capabilities(model_id="qwen3.6-27b")
    assert caps == frozenset({CAP_COMPLETION})


def test_probe_200_confirms_vision():
    assert interpret_probe(status=200, body='{"choices":[]}') is True


def test_probe_400_about_images_confirms_text_only():
    body = '{"error":{"message":"This model does not support image input"}}'
    assert interpret_probe(status=400, body=body) is False


def test_probe_is_inconclusive_on_timeout_or_server_error():
    assert interpret_probe(status=500, body="boom") is None
    assert interpret_probe(status=0, body="") is None


def test_probe_is_inconclusive_on_400_unrelated_to_images():
    body = '{"error":{"message":"max_tokens too large"}}'
    assert interpret_probe(status=400, body=body) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_capabilities.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'archiver.capabilities'`

- [ ] **Step 3: Write minimal implementation**

Crea `archiver/capabilities.py`:

```python
"""Capability dei modelli e parsing della taglia.

Tre gradini di affidabilità decrescente, in quest'ordine:
  declared  - dichiarate dal provider (Ollama >= 0.31 in /api/tags)
  heuristic - dedotte dal nome, per i provider OpenAI-compatible
  probed    - confermate da una richiesta reale, solo dentro il doctor

Il gradino "probed" esiste perché l'euristica ha falsi negativi reali:
qwen3.6-27b servito da vLLM accetta immagini ma non lo dice nel nome.
"""
from __future__ import annotations

import re
from typing import Optional

CAP_COMPLETION = "completion"
CAP_VISION = "vision"
CAP_EMBEDDING = "embedding"

SOURCE_DECLARED = "declared"
SOURCE_HEURISTIC = "heuristic"
SOURCE_PROBED = "probed"

# Un numero conta come taglia solo se seguito da "b" a fine token: in
# "qwen3.6-27b" il 3.6 è la versione e va ignorato.
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", re.IGNORECASE)

_VISION_TOKENS = (
    "llava",
    "moondream",
    "minicpm-v",
    "bakllava",
    "pixtral",
    "internvl",
    "vision",
    "-vl-",
    "-vl:",
    "vl-",
    "qwen2vl",
    "qwen2.5vl",
)

_IMAGE_ERROR_TOKENS = (
    "image",
    "multimodal",
    "vision",
    "image_url",
)


def parse_parameter_size(*texts: str) -> Optional[float]:
    """Prima taglia trovata scorrendo gli argomenti nell'ordine dato."""
    for text in texts:
        if not text:
            continue
        match = _SIZE_RE.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def guess_capabilities(*, model_id: str, root: str = "") -> frozenset[str]:
    """Capability dedotte dal nome, per i provider che non le dichiarano."""
    haystack = f"{model_id} {root}".lower()
    caps = {CAP_COMPLETION}
    if any(token in haystack for token in _VISION_TOKENS):
        caps.add(CAP_VISION)
    return frozenset(caps)


def interpret_probe(*, status: int, body: str) -> Optional[bool]:
    """True = vision confermata, False = text-only confermato, None = non conclusivo.

    Il probe può smentire l'euristica, non può fingere una certezza che non
    ha: un 500 o un timeout non dicono nulla sulle capability del modello.
    """
    if status == 200:
        return True
    if status in (400, 415, 422):
        lowered = body.lower()
        if any(token in lowered for token in _IMAGE_ERROR_TOKENS):
            return False
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_capabilities.py -v`
Expected: PASS, 12 test

- [ ] **Step 5: Commit**

```bash
git add archiver/capabilities.py tests/test_capabilities.py
git commit -m "feat: add model capability detection and parameter size parsing"
```

---

### Task 3: Cache degli esiti del probe

Piccolo modulo di persistenza, isolato perché la Task 4 lo consuma e la Task 14 lo scrive.

**Files:**
- Create: `archiver/probe_cache.py`
- Test: `tests/test_probe_cache.py`

**Interfaces:**
- Consumes: niente
- Produces:
  - `probe_cache_path() -> Path` → `~/.config/amenity-stuff/probe_cache.json` (rispetta `XDG_CONFIG_HOME`)
  - `load_probe_cache(path: Path | None = None) -> dict[tuple[str, str], frozenset[str]]`
  - `save_probe_result(*, url: str, bare_id: str, capabilities: frozenset[str], path: Path | None = None) -> None`
  - `prune_probe_cache(*, known: set[tuple[str, str]], path: Path | None = None) -> None`

- [ ] **Step 1: Write the failing test**

Crea `tests/test_probe_cache.py`:

```python
from __future__ import annotations

import json

from archiver.capabilities import CAP_COMPLETION, CAP_VISION
from archiver.probe_cache import (
    load_probe_cache,
    prune_probe_cache,
    probe_cache_path,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_probe_cache.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'archiver.probe_cache'`

- [ ] **Step 3: Write minimal implementation**

Crea `archiver/probe_cache.py`:

```python
"""Persistenza degli esiti conclusivi del probe delle capability.

Senza questo, il fatto che un modello OpenAI-compatible sia multimodale
andrebbe perso a ogni riavvio e il percorso veloce continuerebbe a fidarsi
dell'euristica sul nome, che su quel caso sbaglia.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_SEP = "|"


def probe_cache_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "amenity-stuff" / "probe_cache.json"


def _resolve(path: Optional[Path]) -> Path:
    return path if path is not None else probe_cache_path()


def _normalise_url(url: str) -> str:
    return url.strip().rstrip("/")


def load_probe_cache(path: Optional[Path] = None) -> dict[tuple[str, str], frozenset[str]]:
    target = _resolve(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[tuple[str, str], frozenset[str]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or _SEP not in key:
            continue
        url, _, bare_id = key.partition(_SEP)
        if not isinstance(value, list):
            continue
        caps = frozenset(v for v in value if isinstance(v, str) and v.strip())
        if url and bare_id and caps:
            out[(url, bare_id)] = caps
    return out


def _write(cache: dict[tuple[str, str], frozenset[str]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {f"{url}{_SEP}{bare}": sorted(caps) for (url, bare), caps in cache.items()}
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def save_probe_result(
    *,
    url: str,
    bare_id: str,
    capabilities: frozenset[str],
    path: Optional[Path] = None,
) -> None:
    target = _resolve(path)
    cache = load_probe_cache(target)
    cache[(_normalise_url(url), bare_id)] = frozenset(capabilities)
    _write(cache, target)


def prune_probe_cache(*, known: set[tuple[str, str]], path: Optional[Path] = None) -> None:
    target = _resolve(path)
    cache = load_probe_cache(target)
    kept = {key: value for key, value in cache.items() if key in known}
    if len(kept) != len(cache):
        _write(kept, target)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_probe_cache.py -v`
Expected: PASS, 7 test

- [ ] **Step 5: Commit**

```bash
git add archiver/probe_cache.py tests/test_probe_cache.py
git commit -m "feat: persist conclusive capability probe results"
```

---

### Task 4: Scoperta dei modelli

Riscrive `discovery.py`. Il parsing dei payload è separato dall'I/O così i test usano i payload **reali** catturati il 2026-08-01 dalla macchina LAN dell'utente senza toccare la rete.

Sostituisce `ProviderInfo` e `DiscoveryResult` attuali. I test esistenti `tests/test_discovery_ds4.py` e `tests/test_discovery_ollama_http.py` vanno riscritti in questa task: verificano il contratto vecchio.

**Files:**
- Modify: `archiver/discovery.py` (riscrittura completa)
- Delete: `tests/test_discovery_ds4.py`, `tests/test_discovery_ollama_http.py`
- Test: `tests/test_discovery.py`

**Interfaces:**
- Consumes: `providers.PROVIDERS`, `providers.KIND_OLLAMA`, `providers.join_model_id`; `capabilities.*`; `probe_cache.load_probe_cache`
- Produces:
  - `ModelInfo` (frozen): `id: str` (prefissato), `provider: str`, `capabilities: frozenset[str]`, `parameter_size_b: float | None`, `context_length: int | None`, `capability_source: str`
  - `ProviderStatus` (frozen): `name: str`, `url: str`, `configured: bool`, `available: bool`, `detail: str`, `models: tuple[ModelInfo, ...]`
  - `DiscoveryResult` (frozen): `providers: tuple[ProviderStatus, ...]`, proprietà `models -> tuple[ModelInfo, ...]`, metodo `status(name: str) -> ProviderStatus | None`
  - `parse_ollama_tags(payload: object) -> tuple[ModelInfo, ...]`
  - `parse_openai_models(payload: object, *, provider_name: str) -> tuple[ModelInfo, ...]`
  - `discover_providers(provider_urls: Mapping[str, str], *, fetch=None, probe_cache=None, timeout_s: float = 2.5) -> DiscoveryResult`

- [ ] **Step 1: Write the failing test**

Elimina i due file obsoleti e crea `tests/test_discovery.py`:

```python
from __future__ import annotations

from archiver.capabilities import (
    CAP_COMPLETION,
    CAP_EMBEDDING,
    CAP_VISION,
    SOURCE_DECLARED,
    SOURCE_HEURISTIC,
    SOURCE_PROBED,
)
from archiver.discovery import (
    discover_providers,
    parse_ollama_tags,
    parse_openai_models,
)

# Payload reale catturato da Ollama 0.31.1 il 2026-08-01.
OLLAMA_TAGS = {
    "models": [
        {
            "name": "qwen3:8b",
            "model": "qwen3:8b",
            "modified_at": "2026-07-03T10:07:48.059344839+02:00",
            "size": 5225388164,
            "digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
            "details": {
                "family": "qwen3",
                "parameter_size": "8.2B",
                "quantization_level": "Q4_K_M",
                "context_length": 40960,
            },
            "capabilities": ["completion", "tools", "thinking"],
        }
    ]
}

# Payload reale catturato da vLLM 0.21.0 il 2026-08-01.
VLLM_MODELS = {
    "object": "list",
    "data": [
        {
            "id": "qwen3.6-27b",
            "object": "model",
            "owned_by": "vllm",
            "root": "/models/Qwen3.6-27B-AWQ-INT4",
            "max_model_len": 131072,
        }
    ],
}


def test_ollama_tags_are_parsed_as_declared_capabilities():
    models = parse_ollama_tags(OLLAMA_TAGS)
    assert len(models) == 1
    m = models[0]
    assert m.id == "ollama:qwen3:8b"
    assert m.provider == "ollama"
    assert m.capability_source == SOURCE_DECLARED
    assert CAP_COMPLETION in m.capabilities
    assert CAP_VISION not in m.capabilities
    assert m.parameter_size_b == 8.2
    assert m.context_length == 40960


def test_ollama_entry_without_capabilities_falls_back_to_heuristic():
    payload = {"models": [{"name": "llava:7b", "details": {"parameter_size": "7B"}}]}
    m = parse_ollama_tags(payload)[0]
    assert m.capability_source == SOURCE_HEURISTIC
    assert CAP_VISION in m.capabilities


def test_openai_models_are_parsed_as_heuristic_with_size_from_root():
    models = parse_openai_models(VLLM_MODELS, provider_name="vllm")
    assert len(models) == 1
    m = models[0]
    assert m.id == "vllm:qwen3.6-27b"
    assert m.provider == "vllm"
    assert m.capability_source == SOURCE_HEURISTIC
    # L'euristica sul nome NON vede la multimodalità: la corregge solo il probe.
    assert CAP_VISION not in m.capabilities
    assert m.parameter_size_b == 27.0
    assert m.context_length == 131072


def test_non_list_payloads_are_rejected_without_raising():
    assert parse_ollama_tags({"models": "nope"}) == ()
    assert parse_ollama_tags("nope") == ()
    assert parse_openai_models({"data": {"id": "x"}}, provider_name="vllm") == ()
    assert parse_openai_models(None, provider_name="vllm") == ()


def test_embedding_models_keep_their_declared_capability():
    payload = {"models": [{"name": "nomic-embed-text:latest",
                           "capabilities": ["embedding"]}]}
    m = parse_ollama_tags(payload)[0]
    assert m.capabilities == frozenset({CAP_EMBEDDING})


def _fetch_from(mapping):
    def fetch(url, *, timeout_s):
        for fragment, payload in mapping.items():
            if fragment in url:
                return payload
        raise ConnectionRefusedError("nothing here")
    return fetch


def test_empty_url_means_not_configured_and_no_request():
    calls = []

    def fetch(url, *, timeout_s):
        calls.append(url)
        return OLLAMA_TAGS

    result = discover_providers(
        {"ollama": "http://ollama.invalid", "vllm": "", "ds4": ""},
        fetch=fetch,
    )
    vllm = result.status("vllm")
    assert vllm.configured is False
    assert vllm.available is False
    assert vllm.models == ()
    assert all("vllm" not in url for url in calls)


def test_unreachable_provider_reports_the_real_reason():
    result = discover_providers(
        {"ollama": "http://ollama.invalid", "vllm": "http://vllm.invalid", "ds4": ""},
        fetch=_fetch_from({"/api/tags": OLLAMA_TAGS}),
    )
    vllm = result.status("vllm")
    assert vllm.configured is True
    assert vllm.available is False
    assert "ConnectionRefusedError" in vllm.detail


def test_models_property_merges_every_available_provider():
    result = discover_providers(
        {"ollama": "http://ollama.invalid", "vllm": "http://vllm.invalid", "ds4": ""},
        fetch=_fetch_from({"/api/tags": OLLAMA_TAGS, "/v1/models": VLLM_MODELS}),
    )
    assert {m.id for m in result.models} == {"ollama:qwen3:8b", "vllm:qwen3.6-27b"}


def test_probe_cache_promotes_capabilities_to_probed():
    cache = {("http://vllm.invalid", "qwen3.6-27b"): frozenset({CAP_COMPLETION, CAP_VISION})}
    result = discover_providers(
        {"ollama": "", "vllm": "http://vllm.invalid", "ds4": ""},
        fetch=_fetch_from({"/v1/models": VLLM_MODELS}),
        probe_cache=cache,
    )
    m = result.status("vllm").models[0]
    assert m.capability_source == SOURCE_PROBED
    assert CAP_VISION in m.capabilities


def test_reachable_provider_with_zero_models_is_available_but_empty():
    result = discover_providers(
        {"ollama": "http://ollama.invalid", "vllm": "", "ds4": ""},
        fetch=_fetch_from({"/api/tags": {"models": []}}),
    )
    ollama = result.status("ollama")
    assert ollama.available is True
    assert ollama.models == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_discovery.py -v`
Expected: FAIL con `ImportError: cannot import name 'parse_ollama_tags' from 'archiver.discovery'`

- [ ] **Step 3: Write minimal implementation**

Sostituisci integralmente `archiver/discovery.py`:

```python
"""Scoperta dei modelli disponibili sui provider configurati.

Una richiesta per provider, le tre in parallelo: il costo peggiore resta
il timeout singolo, non la loro somma. Ollama >= 0.31 dichiara già
capabilities e parameter_size in /api/tags, quindi non serve una richiesta
per modello né una cache delle capability dichiarate.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional
from urllib.request import urlopen

from .capabilities import (
    SOURCE_DECLARED,
    SOURCE_HEURISTIC,
    SOURCE_PROBED,
    guess_capabilities,
    parse_parameter_size,
)
from .providers import KIND_OLLAMA, PROVIDERS, join_model_id, split_model_id


@dataclass(frozen=True)
class ModelInfo:
    id: str
    provider: str
    capabilities: frozenset[str] = frozenset()
    parameter_size_b: Optional[float] = None
    context_length: Optional[int] = None
    capability_source: str = SOURCE_HEURISTIC


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    url: str = ""
    configured: bool = False
    available: bool = False
    detail: str = ""
    models: tuple[ModelInfo, ...] = ()


@dataclass(frozen=True)
class DiscoveryResult:
    providers: tuple[ProviderStatus, ...] = ()

    @property
    def models(self) -> tuple[ModelInfo, ...]:
        out: list[ModelInfo] = []
        for status in self.providers:
            out.extend(status.models)
        return tuple(out)

    def status(self, name: str) -> Optional[ProviderStatus]:
        for status in self.providers:
            if status.name == name:
                return status
        return None


def _get_json(url: str, *, timeout_s: float) -> Any:
    with urlopen(url, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _as_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and value > 0 else None


def parse_ollama_tags(payload: Any) -> tuple[ModelInfo, ...]:
    entries = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return ()
    out: list[ModelInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        declared = entry.get("capabilities")
        if isinstance(declared, list) and declared:
            caps = frozenset(c for c in declared if isinstance(c, str) and c.strip())
            source = SOURCE_DECLARED
        else:
            caps = guess_capabilities(model_id=name)
            source = SOURCE_HEURISTIC
        out.append(
            ModelInfo(
                id=join_model_id("ollama", name),
                provider="ollama",
                capabilities=caps,
                parameter_size_b=parse_parameter_size(
                    str(details.get("parameter_size") or ""), name
                ),
                context_length=_as_int(details.get("context_length")),
                capability_source=source,
            )
        )
    return tuple(out)


def parse_openai_models(payload: Any, *, provider_name: str) -> tuple[ModelInfo, ...]:
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return ()
    out: list[ModelInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        root = entry.get("root") if isinstance(entry.get("root"), str) else ""
        out.append(
            ModelInfo(
                id=join_model_id(provider_name, model_id),
                provider=provider_name,
                capabilities=guess_capabilities(model_id=model_id, root=root),
                parameter_size_b=parse_parameter_size(root, model_id),
                context_length=_as_int(entry.get("max_model_len")),
                capability_source=SOURCE_HEURISTIC,
            )
        )
    return tuple(out)


def _apply_probe_cache(
    models: tuple[ModelInfo, ...],
    *,
    url: str,
    probe_cache: Mapping[tuple[str, str], frozenset[str]],
) -> tuple[ModelInfo, ...]:
    if not probe_cache:
        return models
    key_url = url.rstrip("/")
    out: list[ModelInfo] = []
    for model in models:
        _, bare = split_model_id(model.id)
        cached = probe_cache.get((key_url, bare))
        if cached:
            out.append(
                replace(model, capabilities=cached, capability_source=SOURCE_PROBED)
            )
        else:
            out.append(model)
    return tuple(out)


def _probe_one(
    spec, url: str, *, fetch: Callable[..., Any], timeout_s: float
) -> ProviderStatus:
    url = url.strip()
    if not url:
        return ProviderStatus(name=spec.name, configured=False, detail="non configurato")
    endpoint = url.rstrip("/") + ("/api/tags" if spec.kind == KIND_OLLAMA else "/v1/models")
    try:
        payload = fetch(endpoint, timeout_s=timeout_s)
    except Exception as exc:
        return ProviderStatus(
            name=spec.name,
            url=url,
            configured=True,
            available=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    if spec.kind == KIND_OLLAMA:
        models = parse_ollama_tags(payload)
    else:
        models = parse_openai_models(payload, provider_name=spec.name)
    return ProviderStatus(
        name=spec.name,
        url=url,
        configured=True,
        available=True,
        detail="ok" if models else "raggiungibile, nessun modello",
        models=models,
    )


def discover_providers(
    provider_urls: Mapping[str, str],
    *,
    fetch: Optional[Callable[..., Any]] = None,
    probe_cache: Optional[Mapping[tuple[str, str], frozenset[str]]] = None,
    timeout_s: float = 2.5,
) -> DiscoveryResult:
    fetcher = fetch or _get_json
    cache = probe_cache if probe_cache is not None else {}

    def work(spec) -> ProviderStatus:
        status = _probe_one(
            spec, provider_urls.get(spec.name, "") or "", fetch=fetcher, timeout_s=timeout_s
        )
        if status.models and cache:
            status = replace(
                status, models=_apply_probe_cache(status.models, url=status.url, probe_cache=cache)
            )
        return status

    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as pool:
        statuses = tuple(pool.map(work, PROVIDERS))
    return DiscoveryResult(providers=statuses)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
git rm tests/test_discovery_ds4.py tests/test_discovery_ollama_http.py
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_discovery.py -v
```
Expected: PASS, 10 test

- [ ] **Step 5: Commit**

```bash
git add archiver/discovery.py tests/test_discovery.py
git commit -m "feat: rewrite discovery around ModelInfo and parallel provider probes"
```

---

### Task 5: Ranking dei modelli per ruolo

Riscrive `model_selection.py`, eliminando `_TEXT_PREFER` e `_VISION_PREFER`. `tests/test_model_selection_ds4.py` verifica il contratto vecchio e va sostituito.

**Files:**
- Modify: `archiver/model_selection.py` (riscrittura completa)
- Delete: `tests/test_model_selection_ds4.py`
- Test: `tests/test_model_selection.py`

**Interfaces:**
- Consumes: `discovery.ModelInfo`, `providers.provider_priority`, `providers.split_model_id`, `capabilities.CAP_*`
- Produces:
  - `ROLE_FACTS = "facts"`, `ROLE_CLASSIFY = "classify"`, `ROLE_VISION = "vision"`
  - `CURATED_BIAS: tuple[str, ...]`
  - `size_bucket(size_b: float | None) -> int | None`
  - `rank_models(models: Sequence[ModelInfo], role: str) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Elimina il file obsoleto e crea `tests/test_model_selection.py`:

```python
from __future__ import annotations

from archiver.capabilities import CAP_COMPLETION, CAP_EMBEDDING, CAP_VISION, SOURCE_DECLARED
from archiver.discovery import ModelInfo
from archiver.model_selection import (
    ROLE_CLASSIFY,
    ROLE_FACTS,
    ROLE_VISION,
    rank_models,
    size_bucket,
)


def m(model_id, provider, *, size=None, vision=False, embedding=False):
    caps = {CAP_EMBEDDING} if embedding else {CAP_COMPLETION}
    if vision:
        caps.add(CAP_VISION)
    return ModelInfo(
        id=model_id,
        provider=provider,
        capabilities=frozenset(caps),
        parameter_size_b=size,
        capability_source=SOURCE_DECLARED,
    )


def test_size_buckets_have_five_levels():
    assert size_bucket(1.0) == 0
    assert size_bucket(3.0) == 1
    assert size_bucket(8.2) == 2
    assert size_bucket(13.0) == 3
    assert size_bucket(232.0) == 4
    assert size_bucket(None) is None


def test_provider_priority_comes_before_size():
    models = [
        m("ollama:qwen3:8b", "ollama", size=8.2),
        m("vllm:qwen3.6-27b", "vllm", size=27.0),
        m("ds4:deepseek-v4-flash", "ds4", size=232.0),
    ]
    assert rank_models(models, ROLE_FACTS)[0] == "vllm:qwen3.6-27b"
    assert rank_models(models, ROLE_CLASSIFY)[0] == "vllm:qwen3.6-27b"


def test_ollama_wins_when_vllm_is_absent():
    models = [
        m("ollama:qwen3:8b", "ollama", size=8.2),
        m("ds4:deepseek-v4-flash", "ds4", size=232.0),
    ]
    assert rank_models(models, ROLE_FACTS)[0] == "ollama:qwen3:8b"


def test_within_one_provider_facts_prefers_the_smaller_bucket():
    models = [
        m("ollama:llama3.3:70b", "ollama", size=70.0),
        m("ollama:gemma3:1b", "ollama", size=1.0),
        m("ollama:qwen3:8b", "ollama", size=8.2),
    ]
    assert rank_models(models, ROLE_FACTS) == (
        "ollama:gemma3:1b",
        "ollama:qwen3:8b",
        "ollama:llama3.3:70b",
    )


def test_within_one_provider_classify_prefers_the_5_to_9b_bucket():
    models = [
        m("ollama:gemma3:1b", "ollama", size=1.0),
        m("ollama:qwen3:8b", "ollama", size=8.2),
        m("ollama:llama3.3:70b", "ollama", size=70.0),
    ]
    assert rank_models(models, ROLE_CLASSIFY)[0] == "ollama:qwen3:8b"


def test_curated_bias_breaks_ties_inside_a_bucket():
    # Stessa fascia (2-5B) e stesso provider: decide la lista curata,
    # dove qwen2.5:3b-instruct precede gemma2:2b.
    models = [
        m("ollama:gemma2:2b", "ollama", size=2.0),
        m("ollama:qwen2.5:3b-instruct", "ollama", size=3.0),
    ]
    assert rank_models(models, ROLE_FACTS)[0] == "ollama:qwen2.5:3b-instruct"


def test_curated_bias_matches_on_the_bare_id():
    # Lo stesso modello servito da vLLM deve beneficiare della voce curata
    # scritta senza prefisso.
    models = [
        m("vllm:unknown-3b", "vllm", size=3.0),
        m("vllm:qwen2.5:3b-instruct", "vllm", size=3.0),
    ]
    assert rank_models(models, ROLE_FACTS)[0] == "vllm:qwen2.5:3b-instruct"


def test_unknown_size_goes_last_not_assumed_small():
    models = [
        m("ollama:mystery", "ollama", size=None),
        m("ollama:qwen3:8b", "ollama", size=8.2),
    ]
    assert rank_models(models, ROLE_FACTS)[-1] == "ollama:mystery"


def test_embedding_models_are_excluded_from_text_roles():
    models = [
        m("ollama:nomic-embed-text", "ollama", size=0.1, embedding=True),
        m("ollama:qwen3:8b", "ollama", size=8.2),
    ]
    assert rank_models(models, ROLE_FACTS) == ("ollama:qwen3:8b",)


def test_vision_role_only_returns_models_with_the_capability():
    models = [
        m("ollama:qwen3:8b", "ollama", size=8.2),
        m("vllm:qwen3.6-27b", "vllm", size=27.0, vision=True),
    ]
    assert rank_models(models, ROLE_VISION) == ("vllm:qwen3.6-27b",)


def test_ordering_is_deterministic_for_fully_tied_models():
    models = [
        m("ollama:zzz:3b", "ollama", size=3.0),
        m("ollama:aaa:3b", "ollama", size=3.0),
    ]
    assert rank_models(models, ROLE_FACTS) == ("ollama:aaa:3b", "ollama:zzz:3b")


def test_empty_input_yields_empty_output():
    assert rank_models([], ROLE_FACTS) == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_model_selection.py -v`
Expected: FAIL con `ImportError: cannot import name 'rank_models' from 'archiver.model_selection'`

- [ ] **Step 3: Write minimal implementation**

Sostituisci integralmente `archiver/model_selection.py`:

```python
"""Ordinamento dei modelli candidati, per ruolo.

Sostituisce le tre liste di preferenza hardcoded che vivevano in
model_selection, task_builders e app. L'ordine è deciso dai metadati
reali; la lista curata interviene solo come spareggio dentro una fascia.

Ordine dei criteri:
  1. priorità del provider   (vllm > ollama > ds4)
  2. fascia di taglia        (per ruolo)
  3. posizione in CURATED_BIAS
  4. id completo, alfabetico
"""
from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING

from .capabilities import CAP_COMPLETION, CAP_VISION
from .providers import provider_priority, split_model_id

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import ModelInfo

ROLE_FACTS = "facts"
ROLE_CLASSIFY = "classify"
ROLE_VISION = "vision"

ROLE_CAPABILITY = {
    ROLE_FACTS: CAP_COMPLETION,
    ROLE_CLASSIFY: CAP_COMPLETION,
    ROLE_VISION: CAP_VISION,
}

# Confini fra le fasce, in miliardi di parametri: indici 0..4.
_BUCKET_EDGES = (2.0, 5.0, 9.0, 20.0)
_CLASSIFY_TARGET_BUCKET = 2  # la fascia 5-9B
_UNKNOWN_BUCKET_KEY = 99

# Modelli già provati in passato. Ordina SOLO dentro una fascia, quindi il
# fatto che sia tarata su hardware più limitato non la rende dannosa.
# Manutenuta a mano, con task dedicate: nessun auto-benchmark.
CURATED_BIAS: tuple[str, ...] = (
    "ds4:deepseek-v4-flash",
    "ds4:deepseek-v4-pro",
    "qwen3.6-27b",
    "qwen3:8b",
    "gemma3:1b",
    "qwen2.5:3b-instruct",
    "phi4-mini:latest",
    "phi4-mini",
    "qwen3:4b",
    "qwen3.5:4b",
    "ministral-3:3b",
    "gemma2:2b",
    "qwen2.5:7b",
    "mistral:latest",
    "gemma3:latest",
    "moondream:latest",
    "llava:7b",
    "minicpm-v:latest",
    "bakllava:latest",
)


def size_bucket(size_b: Optional[float]) -> Optional[int]:
    if size_b is None:
        return None
    for index, edge in enumerate(_BUCKET_EDGES):
        if size_b < edge:
            return index
    return len(_BUCKET_EDGES)


def _bucket_key(model: "ModelInfo", role: str) -> int:
    bucket = size_bucket(model.parameter_size_b)
    if bucket is None:
        return _UNKNOWN_BUCKET_KEY
    if role == ROLE_CLASSIFY:
        return abs(bucket - _CLASSIFY_TARGET_BUCKET)
    return bucket


def _curated_key(model_id: str) -> int:
    if model_id in CURATED_BIAS:
        return CURATED_BIAS.index(model_id)
    _, bare = split_model_id(model_id)
    if bare in CURATED_BIAS:
        return CURATED_BIAS.index(bare)
    return len(CURATED_BIAS)


def rank_models(models: Sequence["ModelInfo"], role: str) -> tuple[str, ...]:
    required = ROLE_CAPABILITY.get(role, CAP_COMPLETION)
    eligible = [model for model in models if required in model.capabilities]
    eligible.sort(
        key=lambda model: (
            provider_priority(model.provider),
            _bucket_key(model, role),
            _curated_key(model.id),
            model.id,
        )
    )
    return tuple(model.id for model in eligible)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
git rm tests/test_model_selection_ds4.py
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_model_selection.py -v
```
Expected: PASS, 12 test

- [ ] **Step 5: Commit**

```bash
git add archiver/model_selection.py tests/test_model_selection.py
git commit -m "feat: rank models by metadata instead of three hardcoded lists"
```

---

### Task 6: Backend OpenAI-compatible condiviso, con immagini

`Ds4Backend` diventa `OpenAICompatBackend`, usato da ds4 e vLLM. Aggiunge il supporto immagini e sposta il quirk `reasoning_effort` nel registry. `tests/test_openai_client.py` va aggiornato al nuovo nome e ai nuovi comportamenti.

**Files:**
- Modify: `archiver/openai_client.py`
- Modify: `tests/test_openai_client.py`

**Interfaces:**
- Consumes: `providers.ProviderSpec`, `providers.provider_by_name`; `llm_backend.BaseLLMBackend`, `llm_backend.LLMResponse`
- Produces:
  - `OpenAICompatBackend(base_url: str, spec: ProviderSpec | None = None)` con `generate(...) -> LLMResponse` (stessa firma del protocollo `LLMBackend`)
  - `mime_from_b64(b64: str) -> str`
  - Costante `MAX_TOKENS = 8000`

- [ ] **Step 1: Write the failing test**

Sostituisci `tests/test_openai_client.py`:

```python
from __future__ import annotations

import base64
import json

from archiver import openai_client
from archiver.openai_client import MAX_TOKENS, OpenAICompatBackend, mime_from_b64
from archiver.providers import provider_by_name

PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)


def _capture(monkeypatch, response):
    captured = {}

    def fake_post(url, payload, *, timeout_s):
        captured["url"] = url
        captured["payload"] = payload
        return response

    monkeypatch.setattr(openai_client, "_post_json", fake_post)
    return captured


def _ok(text="hello"):
    return {
        "model": "m",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
    }


def test_mime_is_sniffed_from_magic_bytes():
    assert mime_from_b64(PNG_1X1) == "image/png"
    jpeg = base64.b64encode(b"\xff\xd8\xff\xe0somejunk").decode("ascii")
    assert mime_from_b64(jpeg) == "image/jpeg"
    assert mime_from_b64("!!!not base64!!!") == "image/png"


def test_images_are_sent_as_openai_image_url_parts(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    backend = OpenAICompatBackend("http://vllm.invalid:8000", provider_by_name("vllm"))

    result = backend.generate(prompt="describe", model="qwen3.6-27b", images_b64=[PNG_1X1])

    assert result.success
    content = captured["payload"]["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_text_only_prompt_stays_a_plain_string(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    backend = OpenAICompatBackend("http://vllm.invalid:8000", provider_by_name("vllm"))

    backend.generate(prompt="hi", model="m")

    assert captured["payload"]["messages"][0]["content"] == "hi"


def test_reasoning_effort_is_sent_only_for_ds4(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    OpenAICompatBackend("http://ds4.invalid", provider_by_name("ds4")).generate(
        prompt="hi", model="deepseek-v4-flash", think=False
    )
    assert captured["payload"]["reasoning_effort"] == "low"

    captured = _capture(monkeypatch, _ok())
    OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm")).generate(
        prompt="hi", model="qwen3.6-27b", think=False
    )
    assert "reasoning_effort" not in captured["payload"]


def test_max_tokens_is_capped_by_declared_context_length(monkeypatch):
    captured = _capture(monkeypatch, _ok())
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))

    backend.generate(prompt="hi", model="m", max_model_len=2048)

    assert captured["payload"]["max_tokens"] == 2048

    captured = _capture(monkeypatch, _ok())
    backend.generate(prompt="hi", model="m", max_model_len=131072)
    assert captured["payload"]["max_tokens"] == MAX_TOKENS


def test_truncation_by_length_is_reported_as_an_error(monkeypatch):
    _capture(monkeypatch, {
        "choices": [{"message": {"content": "part"}, "finish_reason": "length"}]
    })
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "truncated" in (result.error or "")


def test_reasoning_field_is_never_read_as_content(monkeypatch):
    # Verificato sul campo: qwen3.6-27b riempie "reasoning" lasciando
    # "content" a null finché non ha finito di ragionare.
    _capture(monkeypatch, {
        "choices": [{
            "message": {"content": None, "reasoning": "The user has provided an image"},
            "finish_reason": "stop",
        }]
    })
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "empty content" in (result.error or "")


def test_server_error_payload_becomes_an_error_response(monkeypatch):
    _capture(monkeypatch, {"error": {"message": "model not found"}})
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "model not found" in (result.error or "")


def test_transport_exception_becomes_an_error_response(monkeypatch):
    def boom(url, payload, *, timeout_s):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(openai_client, "_post_json", boom)
    backend = OpenAICompatBackend("http://vllm.invalid", provider_by_name("vllm"))
    result = backend.generate(prompt="hi", model="m")
    assert not result.success
    assert "ConnectionRefusedError" in (result.error or "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_openai_client.py -v`
Expected: FAIL con `ImportError: cannot import name 'OpenAICompatBackend' from 'archiver.openai_client'`

- [ ] **Step 3: Write minimal implementation**

Sostituisci integralmente `archiver/openai_client.py`:

```python
"""Backend OpenAI-compatible, condiviso da ds4 e vLLM.

Implementa il protocollo LLMBackend su POST /v1/chat/completions. Legge
SOLO message.content, mai i campi di ragionamento: i modelli reasoning
riempiono un campo separato lasciando content a null finché non hanno
finito, e prenderlo per risposta significherebbe archiviare il monologo
del modello invece del suo output.

Le differenze fra provider dello stesso tipo (oggi: solo ds4 accetta
reasoning_effort) vivono nel registry, non qui.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Optional
from urllib.request import Request, urlopen

from .llm_backend import BaseLLMBackend, LLMResponse
from .providers import ProviderSpec

MAX_TOKENS = 8000


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def mime_from_b64(b64: str) -> str:
    """Deduce il MIME dai magic bytes; default png se illeggibile."""
    try:
        head = base64.b64decode(b64[:32] + "==", validate=False)
    except Exception:
        return "image/png"
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"GIF8"):
        return "image/gif"
    return "image/png"


class OpenAICompatBackend(BaseLLMBackend):
    """Backend per qualunque server che parli l'API chat-completions di OpenAI.

    Usage:
        backend = OpenAICompatBackend(url, provider_by_name("vllm"))
        response = backend.generate(prompt="Hello", model="qwen3.6-27b")
    """

    def __init__(self, base_url: str, spec: Optional[ProviderSpec] = None):
        super().__init__(base_url)
        self.spec = spec

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        timeout_s: float = 120.0,
        images_b64: Optional[list[str]] = None,
        response_format: str | dict[str, Any] | None = None,
        think: bool | str | None = None,
        keep_alive: str | int | None = None,
        options: Optional[dict[str, Any]] = None,
        max_model_len: Optional[int] = None,
    ) -> LLMResponse:
        if images_b64:
            content: Any = [{"type": "text", "text": prompt}]
            for b64 in images_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_from_b64(b64)};base64,{b64}"},
                    }
                )
        else:
            content = prompt

        max_tokens = MAX_TOKENS
        if isinstance(max_model_len, int) and 0 < max_model_len < MAX_TOKENS:
            max_tokens = max_model_len

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_tokens": max_tokens,
        }
        if think is False and self.spec is not None and self.spec.sends_reasoning_effort:
            payload["reasoning_effort"] = "low"
        temperature = (options or {}).get("temperature")
        if isinstance(temperature, (int, float)):
            payload["temperature"] = temperature
        # response_format non è imposto dal server; keep_alive e num_predict
        # sono specifici di Ollama. La forma JSON è garantita dai prompt più
        # il normalizer e la riparazione JSON già esistenti.

        try:
            data = _post_json(f"{self.base_url}/v1/chat/completions", payload, timeout_s=timeout_s)
        except Exception as exc:
            return LLMResponse(text="", error=f"{type(exc).__name__}: {exc}", done=False)

        if not isinstance(data, dict):
            return LLMResponse(text="", error="openai-compat: malformed response", done=False)

        err = data.get("error")
        if err:
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return LLMResponse(text="", error=str(msg), done=False)

        try:
            choice = data["choices"][0]
            content_out = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return LLMResponse(text="", error="openai-compat: malformed response", done=False)

        if choice.get("finish_reason") == "length":
            return LLMResponse(text="", error="openai-compat: output truncated by max_tokens", done=False)
        if not content_out.strip():
            return LLMResponse(text="", error="openai-compat: empty content", done=False)
        return LLMResponse(text=content_out, model=data.get("model") or model, done=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_openai_client.py -v`
Expected: PASS, 10 test

- [ ] **Step 5: Commit**

```bash
git add archiver/openai_client.py tests/test_openai_client.py
git commit -m "feat: share one OpenAI-compatible backend with image support"
```

---

### Task 7: Routing per prefisso

`llm_router` smette di ragionare per casi speciali. `tests/test_llm_router.py` va riscritto: verifica la firma vecchia.

**Files:**
- Modify: `archiver/llm_router.py` (riscrittura completa)
- Modify: `tests/test_llm_router.py`

**Interfaces:**
- Consumes: `providers.split_model_id`, `providers.KIND_OLLAMA`; `ollama_client.OllamaGenerateResult`, `ollama_client.OllamaBackend`; `openai_client.OpenAICompatBackend`
- Produces:
  - `generate(*, model: str, prompt: str, provider_urls: Mapping[str, str], timeout_s: float = 120.0, images_b64=None, response_format=None, think=None, keep_alive=None, options=None, max_model_len: int | None = None) -> OllamaGenerateResult`
  - `generate_with_image_file(*, model: str, prompt: str, image_path: str, provider_urls: Mapping[str, str], timeout_s: float = 180.0) -> OllamaGenerateResult`

Il tipo di ritorno resta `OllamaGenerateResult` per non propagare il cambiamento fino a `analyzer.py` e `normalizer.py`, che lo consumano oggi.

- [ ] **Step 1: Write the failing test**

Sostituisci `tests/test_llm_router.py`:

```python
from __future__ import annotations

import base64

from archiver import llm_router
from archiver.llm_backend import LLMResponse

URLS = {
    "ollama": "http://ollama.invalid:11434",
    "vllm": "http://vllm.invalid:8000",
    "ds4": "",
}


class _Spy:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, base_url, spec=None):
        self.calls.append({"base_url": base_url, "spec": spec})
        return self

    def generate(self, **kwargs):
        self.calls[-1].update(kwargs)
        return self.response


def test_bare_legacy_id_routes_to_ollama(monkeypatch):
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OllamaBackend", spy)

    result = llm_router.generate(model="qwen3:8b", prompt="hi", provider_urls=URLS)

    assert result.response == "ok"
    assert spy.calls[0]["base_url"] == URLS["ollama"]
    assert spy.calls[0]["model"] == "qwen3:8b"


def test_prefixed_ollama_id_strips_the_prefix_before_the_call(monkeypatch):
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OllamaBackend", spy)

    llm_router.generate(model="ollama:qwen3:8b", prompt="hi", provider_urls=URLS)

    assert spy.calls[0]["model"] == "qwen3:8b"


def test_vllm_id_routes_to_the_openai_backend_with_its_spec(monkeypatch):
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OpenAICompatBackend", spy)

    result = llm_router.generate(
        model="vllm:qwen3.6-27b", prompt="hi", provider_urls=URLS
    )

    assert result.response == "ok"
    assert spy.calls[0]["base_url"] == URLS["vllm"]
    assert spy.calls[0]["spec"].name == "vllm"
    assert spy.calls[0]["model"] == "qwen3.6-27b"


def test_unconfigured_provider_fails_explicitly_instead_of_falling_back():
    result = llm_router.generate(model="ds4:whatever", prompt="hi", provider_urls=URLS)
    assert result.done is False
    assert "ds4" in (result.error or "")
    assert "configurat" in (result.error or "")


def test_image_file_goes_to_ollama_as_base64(monkeypatch, tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OllamaBackend", spy)

    llm_router.generate_with_image_file(
        model="ollama:llava:7b", prompt="what", image_path=str(png), provider_urls=URLS
    )

    sent = spy.calls[0]["images_b64"][0]
    assert base64.b64decode(sent).startswith(b"\x89PNG")


def test_image_file_now_works_on_openai_compat_providers(monkeypatch, tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    spy = _Spy(LLMResponse(text="ok"))
    monkeypatch.setattr(llm_router, "OpenAICompatBackend", spy)

    result = llm_router.generate_with_image_file(
        model="vllm:qwen3.6-27b", prompt="what", image_path=str(png), provider_urls=URLS
    )

    assert result.done is True
    assert spy.calls[0]["images_b64"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_llm_router.py -v`
Expected: FAIL con `TypeError: generate() got an unexpected keyword argument 'provider_urls'`

- [ ] **Step 3: Write minimal implementation**

Sostituisci integralmente `archiver/llm_router.py`:

```python
"""Instrada le chiamate LLM al backend giusto, in base al prefisso del model-id.

Convenzione: ogni model-id porta con sé il provider ("ollama:", "vllm:",
"ds4:"). Il prefisso viaggia ovunque — candidati, config, model_used in
cache, UI — quindi non serve altro stato per sapere da dove viene un
modello. Gli id senza prefisso noto vengono da config scritte prima della
0.12.0 e valgono come Ollama.

Il layer è stateless: un backend per chiamata, nessuno stato mutabile
condiviso, così una futura scansione parallela non dovrà toccarlo.
"""
from __future__ import annotations

import base64
from typing import Any, Mapping, Optional

from .llm_backend import LLMResponse
from .ollama_client import OllamaBackend, OllamaGenerateResult
from .openai_client import OpenAICompatBackend
from .providers import KIND_OLLAMA, split_model_id


def _to_legacy(response: LLMResponse, *, model: str) -> OllamaGenerateResult:
    return OllamaGenerateResult(
        response=response.text,
        model=model,
        done=response.done,
        error=response.error,
    )


def _resolve(model: str, provider_urls: Mapping[str, str]):
    spec, bare_id = split_model_id(model)
    url = (provider_urls.get(spec.name) or "").strip()
    if not url:
        return None, spec, bare_id
    if spec.kind == KIND_OLLAMA:
        return OllamaBackend(url), spec, bare_id
    return OpenAICompatBackend(url, spec), spec, bare_id


def generate(
    *,
    model: str,
    prompt: str,
    provider_urls: Mapping[str, str],
    timeout_s: float = 120.0,
    images_b64: Optional[list[str]] = None,
    response_format: str | dict[str, Any] | None = None,
    think: bool | str | None = None,
    keep_alive: str | int | None = None,
    options: Optional[dict[str, Any]] = None,
    max_model_len: Optional[int] = None,
) -> OllamaGenerateResult:
    backend, spec, bare_id = _resolve(model, provider_urls)
    if backend is None:
        return OllamaGenerateResult(
            response="", error=f"{spec.name}: endpoint non configurato", done=False
        )
    kwargs: dict[str, Any] = dict(
        prompt=prompt,
        model=bare_id,
        timeout_s=timeout_s,
        images_b64=images_b64,
        response_format=response_format,
        think=think,
        keep_alive=keep_alive,
        options=options,
    )
    if spec.kind != KIND_OLLAMA:
        kwargs["max_model_len"] = max_model_len
    return _to_legacy(backend.generate(**kwargs), model=model)


def generate_with_image_file(
    *,
    model: str,
    prompt: str,
    image_path: str,
    provider_urls: Mapping[str, str],
    timeout_s: float = 180.0,
) -> OllamaGenerateResult:
    with open(image_path, "rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("ascii")
    return generate(
        model=model,
        prompt=prompt,
        provider_urls=provider_urls,
        timeout_s=timeout_s,
        images_b64=[b64],
        keep_alive="5m",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_llm_router.py -v`
Expected: PASS, 6 test

- [ ] **Step 5: Commit**

```bash
git add archiver/llm_router.py tests/test_llm_router.py
git commit -m "feat: route llm calls by provider prefix with explicit urls"
```

---

### Task 8: Propagazione di `provider_urls` nell'analyzer

Modifica meccanica ma estesa: `AnalysisConfig` perde `ollama_base_url` e `ds4_base_url` in favore di `provider_urls`, e ogni chiamata a `generate` viene aggiornata. `tests/test_analyzer_routing.py` e `tests/test_normalizer_routing.py` vanno adeguati.

**Files:**
- Modify: `archiver/analyzer.py` (`AnalysisConfig` a `:47`, `_repair_json_dict_via_llm` a `:226`, `_classify_from_text` a `:293`, `_try_text_models` a `:479`, e ogni altra occorrenza di `base_url=`/`ds4_base_url=`)
- Modify: `archiver/normalizer.py` (stesse sostituzioni)
- Modify: `tests/test_analyzer_routing.py`, `tests/test_normalizer_routing.py`

**Interfaces:**
- Consumes: `llm_router.generate` (nuova firma dalla Task 7)
- Produces: `AnalysisConfig.provider_urls: dict[str, str]` (default `{}` normalizzato in `__post_init__`)

- [ ] **Step 1: Write the failing test**

Sostituisci `tests/test_analyzer_routing.py`:

```python
from __future__ import annotations

import json

from archiver import analyzer
from archiver.analyzer import AnalysisConfig, _classify_from_text
from archiver.ollama_client import OllamaGenerateResult
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines

_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)

URLS = {"ollama": "http://ollama.invalid:11434", "vllm": "http://vllm.invalid:8000", "ds4": ""}


def test_analysis_config_defaults_to_empty_provider_urls():
    assert AnalysisConfig().provider_urls == {}


def test_analysis_config_no_longer_exposes_flat_urls():
    assert not hasattr(AnalysisConfig(), "ds4_base_url")
    assert not hasattr(AnalysisConfig(), "ollama_base_url")


def _fake_generate(captured, payload_text):
    def fake(**kwargs):
        captured.update(kwargs)
        return OllamaGenerateResult(response=payload_text, model=kwargs["model"], done=True)

    return fake


def test_classify_forwards_provider_urls_to_the_router(monkeypatch):
    captured = {}
    out = json.dumps({"category": "unknown", "reference_year": None, "proposed_name": "doc"})
    monkeypatch.setattr(analyzer, "generate", _fake_generate(captured, out))

    _classify_from_text(
        model="vllm:qwen3.6-27b",
        content="some text",
        filename="a.pdf",
        mtime_iso="2026-01-01T00:00:00",
        provider_urls=URLS,
        reference_year_hint=None,
        category_hint=None,
        output_language="en",
        taxonomy=_TAXONOMY,
    )

    assert captured["provider_urls"] == URLS
    assert captured["model"] == "vllm:qwen3.6-27b"
```

Adegua allo stesso modo `tests/test_normalizer_routing.py`: sostituisci ogni `base_url=...`/`ds4_base_url=...` con `provider_urls=URLS` e ogni assert su quei parametri con l'equivalente su `provider_urls`.

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_analyzer_routing.py tests/test_normalizer_routing.py -v`
Expected: FAIL con `AttributeError`/`TypeError` sui parametri `base_url`

- [ ] **Step 3: Write minimal implementation**

In `archiver/analyzer.py`, sostituisci i due campi in `AnalysisConfig`:

```python
@dataclass(frozen=True)
class AnalysisConfig:
    text_model: str = "gemma3:1b"
    vision_model: str = "moondream:latest"
    text_models: tuple[str, ...] = ()
    vision_models: tuple[str, ...] = ()
    provider_urls: dict[str, str] = None  # type: ignore[assignment]
    output_language: str = "auto"  # auto | it | en
    taxonomy: Taxonomy = _DEFAULT_TAXONOMY
    filename_separator: str = "space"  # space | underscore | dash
    ocr_mode: str = "balanced"  # fast | balanced | high

    def __post_init__(self) -> None:
        if self.provider_urls is None:
            object.__setattr__(self, "provider_urls", {})
```

Poi, in `analyzer.py` e `normalizer.py`, applica queste sostituzioni meccaniche ovunque:

- nelle firme di funzione: `base_url: str, ds4_base_url: str = ""` → `provider_urls: Mapping[str, str]`
- nelle chiamate a `generate(...)`: `base_url=..., ds4_base_url=...` → `provider_urls=provider_urls`
- ai call site che leggono la config: `base_url=cfg.ollama_base_url, ds4_base_url=cfg.ds4_base_url` → `provider_urls=cfg.provider_urls`
- aggiungi `from typing import Mapping` dove serve

Trova ogni occorrenza con:

```bash
TOKENSAVE_DISABLE_GREP_HOOK=1 grep -rn "ds4_base_url\|ollama_base_url" archiver/
```

- [ ] **Step 4: Run test to verify it passes**

```bash
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v
TOKENSAVE_DISABLE_GREP_HOOK=1 grep -rn "ds4_base_url" archiver/analyzer.py archiver/normalizer.py
```
Expected: pytest PASS; il grep non deve restituire nulla per questi due file

- [ ] **Step 5: Commit**

```bash
git add archiver/analyzer.py archiver/normalizer.py tests/test_analyzer_routing.py tests/test_normalizer_routing.py
git commit -m "refactor: pass provider urls as a mapping through the analysis layer"
```

---

### Task 9: Configurazione a tre URL, con migrazione

**Files:**
- Modify: `archiver/config.py`, `archiver/settings.py`
- Test: `tests/test_config_migration.py`

**Interfaces:**
- Consumes: `providers.PROVIDER_NAMES`, `providers.default_provider_urls`, `providers.split_model_id`, `providers.join_model_id`
- Produces:
  - `AppConfig.providers: dict[str, str]` (sostituisce `ollama_base_url` e `ds4_base_url`; sparisce `vision_model_fallback`)
  - `Settings.providers: dict[str, str]` (idem)
  - `config.migrate_model_id(value: str) -> str`

- [ ] **Step 1: Write the failing test**

Crea `tests/test_config_migration.py`:

```python
from __future__ import annotations

import json

from archiver.config import AppConfig, load_config, migrate_model_id, save_config


def _write(tmp_path, monkeypatch, data):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "amenity-stuff" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_defaults_have_three_providers_and_no_real_hostnames():
    cfg = AppConfig()
    assert set(cfg.providers) == {"ollama", "vllm", "ds4"}
    assert cfg.providers["ollama"] == "http://localhost:11434"
    assert cfg.providers["vllm"] == ""
    assert cfg.providers["ds4"] == ""


def test_flat_urls_are_migrated_into_the_providers_mapping(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {
        "ollama_base_url": "http://box.invalid:11434",
        "ds4_base_url": "http://box.invalid:9000",
    })
    cfg = load_config()
    assert cfg.providers["ollama"] == "http://box.invalid:11434"
    assert cfg.providers["ds4"] == "http://box.invalid:9000"
    assert cfg.providers["vllm"] == ""


def test_bare_model_ids_are_migrated_to_the_ollama_prefix():
    assert migrate_model_id("qwen3:8b") == "ollama:qwen3:8b"
    assert migrate_model_id("ds4:deepseek-v4-flash") == "ds4:deepseek-v4-flash"
    assert migrate_model_id("auto") == "auto"
    assert migrate_model_id("") == "auto"


def test_pinned_models_in_an_old_config_are_migrated(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {
        "facts_model": "gemma3:1b",
        "classify_model": "ds4:deepseek-v4-flash",
        "vision_model": "auto",
    })
    cfg = load_config()
    assert cfg.facts_model == "ollama:gemma3:1b"
    assert cfg.classify_model == "ds4:deepseek-v4-flash"
    assert cfg.vision_model == "auto"


def test_legacy_vision_model_fallback_is_dropped(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"vision_model_fallback": "llava:7b"})
    cfg = load_config()
    assert not hasattr(cfg, "vision_model_fallback")


def test_legacy_text_model_migration_still_works(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"text_model": "gemma3:1b"})
    cfg = load_config()
    assert cfg.facts_model == "ollama:gemma3:1b"
    assert cfg.classify_model == "ollama:gemma3:1b"


def test_new_format_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_config(AppConfig(providers={"ollama": "http://a.invalid", "vllm": "http://b.invalid", "ds4": ""}))
    cfg = load_config()
    assert cfg.providers["vllm"] == "http://b.invalid"


def test_unknown_provider_keys_are_ignored(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"providers": {"ollama": "http://a.invalid", "bogus": "x"}})
    cfg = load_config()
    assert set(cfg.providers) == {"ollama", "vllm", "ds4"}


def test_missing_file_yields_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert load_config().providers["ollama"] == "http://localhost:11434"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_config_migration.py -v`
Expected: FAIL con `ImportError: cannot import name 'migrate_model_id'`

- [ ] **Step 3: Write minimal implementation**

In `archiver/config.py`: rimuovi i campi `ds4_base_url`, `ollama_base_url` e `vision_model_fallback` da `AppConfig`, aggiungi `providers`, aggiungi `migrate_model_id`, e nel corpo di `load_config()` inserisci la migrazione. Le parti che non riguardano i provider (lingua, tassonomie, separatore, OCR, cartella undated) restano identiche.

```python
from .providers import PROVIDER_NAMES, default_provider_urls, split_model_id


@dataclass(frozen=True)
class AppConfig:
    last_archive_root: Optional[str] = None
    last_source_root: Optional[str] = None
    output_language: str = "auto"  # auto | it | en
    taxonomies: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]
    facts_model: str = "auto"
    classify_model: str = "auto"
    vision_model: str = "auto"
    filename_separator: str = "space"  # space | underscore | dash
    ocr_mode: str = "balanced"  # fast | balanced | high
    undated_folder_name: str = "undated"
    providers: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.taxonomies is None:
            object.__setattr__(self, "taxonomies", {})
        merged = default_provider_urls()
        if self.providers:
            for name, url in self.providers.items():
                if name in merged and isinstance(url, str):
                    merged[name] = url.strip()
        object.__setattr__(self, "providers", merged)


def migrate_model_id(value: str) -> str:
    """Gli id nudi delle config < 0.12.0 significavano Ollama."""
    value = (value or "").strip()
    if not value or value == "auto":
        return "auto"
    spec, bare = split_model_id(value)
    return spec.prefix + bare
```

Dentro `load_config()`, sostituisci il blocco che legge `ds4_base_url`/`ollama_base_url` con:

```python
    # providers: formato nuovo, con fallback sulle due chiavi piatte < 0.12.0
    providers = default_provider_urls()
    providers_raw = data.get("providers")
    if isinstance(providers_raw, dict):
        for name, url in providers_raw.items():
            if name in PROVIDER_NAMES and isinstance(url, str) and url.strip():
                providers[name] = url.strip()
    else:
        legacy_ollama = data.get("ollama_base_url")
        legacy_ds4 = data.get("ds4_base_url")
        if isinstance(legacy_ollama, str) and legacy_ollama.strip():
            providers["ollama"] = legacy_ollama.strip()
        if isinstance(legacy_ds4, str) and legacy_ds4.strip():
            providers["ds4"] = legacy_ds4.strip()
    kwargs["providers"] = providers
```

e avvolgi le assegnazioni dei tre modelli in `migrate_model_id(...)`:

```python
    if isinstance(legacy_text_model, str) and legacy_text_model.strip():
        legacy = migrate_model_id(legacy_text_model)
        kwargs["facts_model"] = legacy
        kwargs["classify_model"] = legacy
    if isinstance(facts_model, str) and facts_model.strip():
        kwargs["facts_model"] = migrate_model_id(facts_model)
    if isinstance(classify_model, str) and classify_model.strip():
        kwargs["classify_model"] = migrate_model_id(classify_model)
    if isinstance(vision_model, str) and vision_model.strip():
        kwargs["vision_model"] = migrate_model_id(vision_model)
```

Rimuovi ogni lettura di `vision_model_fallback`.

In `archiver/settings.py`: rimuovi `ds4_base_url`, `ollama_base_url` e `vision_model_fallback`; aggiungi `providers: dict[str, str] = None` con la stessa normalizzazione in `__post_init__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_config_migration.py -v`
Expected: PASS, 9 test

- [ ] **Step 5: Commit**

```bash
git add archiver/config.py archiver/settings.py tests/test_config_migration.py
git commit -m "feat: store provider endpoints as a mapping and migrate old configs"
```

---

### Task 10: Schermata settings a tre URL

**Files:**
- Modify: `archiver/settings_screen.py`
- Modify: `tests/test_settings_screen_result.py`

**Interfaces:**
- Consumes: `providers.PROVIDERS`, `discovery.ModelInfo`, `model_selection.rank_models`
- Produces: `SettingsResult` con `providers: dict[str, str]`, senza `ds4_base_url`, `ollama_base_url`, `vision_model_fallback`. Il costruttore di `SettingsScreen` prende `providers: dict[str, str]` e `available_models: tuple[ModelInfo, ...]` (non più `tuple[str, ...]`).

- [ ] **Step 1: Write the failing test**

Sostituisci `tests/test_settings_screen_result.py`:

```python
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from archiver.settings_screen import SettingsResult


def test_result_carries_a_providers_mapping():
    names = {f.name for f in fields(SettingsResult)}
    assert "providers" in names
    assert "ds4_base_url" not in names
    assert "ollama_base_url" not in names
    assert "vision_model_fallback" not in names


def test_result_is_constructible_with_three_providers():
    result = SettingsResult(
        output_language="it",
        taxonomies={},
        facts_model="auto",
        classify_model="auto",
        vision_model="auto",
        filename_separator="space",
        ocr_mode="balanced",
        undated_folder_name="undated",
        archive_root=Path("/tmp/archive"),
        providers={"ollama": "http://a.invalid", "vllm": "", "ds4": ""},
    )
    assert set(result.providers) == {"ollama", "vllm", "ds4"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_settings_screen_result.py -v`
Expected: FAIL con `TypeError: __init__() got an unexpected keyword argument 'providers'`

- [ ] **Step 3: Write minimal implementation**

In `archiver/settings_screen.py`:

1. `SettingsResult`: sostituisci `ds4_base_url` e `ollama_base_url` con `providers: dict[str, str]`; rimuovi `vision_model_fallback`.
2. CSS: sostituisci le regole `#ds4_label`/`#ds4_url`/`#ollama_label`/`#ollama_url` con una regola generica, dato che gli id ora sono generati dal registry:

```css
    .provider_label { height: auto; padding: 1 0 0 0; }
    .provider_url { height: 3; }
```

3. `__init__`: prendi `providers: dict[str, str]` e `available_models: tuple[ModelInfo, ...]`; rimuovi `vision_model_fallback`. Le tre liste di opzioni si costruiscono con il ranking invece che con i filtri `_filter_text_models`/`_filter_vision_models`, che vanno eliminati:

```python
from .model_selection import ROLE_CLASSIFY, ROLE_FACTS, ROLE_VISION, rank_models
from .providers import PROVIDER_NAMES, PROVIDERS

        self._providers = {name: (providers or {}).get(name, "") for name in PROVIDER_NAMES}
        self._facts_options = ("auto",) + rank_models(available_models, ROLE_FACTS)
        self._classify_options = ("auto",) + rank_models(available_models, ROLE_CLASSIFY)
        self._vision_options = ("auto",) + rank_models(available_models, ROLE_VISION)
```

4. `compose()`: un `Static` + `Input` per ogni provider del registry, al posto dei due blocchi fissi:

```python
        for spec in PROVIDERS:
            placeholder = spec.default_url or "vuoto = disabilitato"
            yield Static(f"{spec.name} endpoint ({placeholder}):",
                         classes="provider_label", id=f"{spec.name}_label")
            yield Input(value=self._providers.get(spec.name, ""),
                        placeholder=placeholder,
                        classes="provider_url", id=f"{spec.name}_url")
```

5. Sostituisci `_current_ds4_url()` e `_current_ollama_url()` con:

```python
    def _current_provider_urls(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in PROVIDERS:
            try:
                value = self.query_one(f"#{spec.name}_url", Input).value.strip()
            except Exception:
                value = self._providers.get(spec.name, "")
            out[spec.name] = value or spec.default_url
        return out
```

6. Rimuovi la voce "Vision fallback" dalla lista delle opzioni renderizzate e dalla logica di `_activate_option`, e passa `providers=self._current_provider_urls()` alla costruzione di `SettingsResult`.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v`
Expected: PASS (la suite intera; `app.py` non compila ancora contro le nuove firme ma non è importato dai test — se lo fosse, questa task va unita alla Task 11)

- [ ] **Step 5: Commit**

```bash
git add archiver/settings_screen.py tests/test_settings_screen_result.py
git commit -m "tui: configure three provider endpoints and drop the vision fallback"
```

---

### Task 11: Cablaggio dell'app

Collega tutto: `task_builders`, `ui_status`, `app.py`. Elimina `_ordered_classify_models`.

**Files:**
- Modify: `archiver/task_builders.py`, `archiver/ui_status.py`, `archiver/app.py`, `archiver/__main__.py`
- Test: `tests/test_task_builders.py`

**Interfaces:**
- Consumes: `discovery.DiscoveryResult`, `model_selection.rank_models`, `settings.Settings.providers`
- Produces: `task_builders.build_analysis_config(*, settings, discovery, taxonomy) -> AnalysisConfig` (firma invariata, implementazione nuova)

- [ ] **Step 1: Write the failing test**

Crea `tests/test_task_builders.py`:

```python
from __future__ import annotations

from archiver.capabilities import CAP_COMPLETION, CAP_VISION, SOURCE_DECLARED
from archiver.discovery import DiscoveryResult, ModelInfo, ProviderStatus
from archiver.settings import Settings
from archiver.task_builders import build_analysis_config
from archiver.taxonomy import DEFAULT_TAXONOMY_LINES, parse_taxonomy_lines
from pathlib import Path

_TAXONOMY, _ = parse_taxonomy_lines(DEFAULT_TAXONOMY_LINES)


def _model(model_id, provider, size, *, vision=False):
    caps = {CAP_COMPLETION} | ({CAP_VISION} if vision else set())
    return ModelInfo(
        id=model_id, provider=provider, capabilities=frozenset(caps),
        parameter_size_b=size, capability_source=SOURCE_DECLARED,
    )


def _discovery():
    return DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://vllm.invalid", configured=True, available=True,
                       models=(_model("vllm:qwen3.6-27b", "vllm", 27.0, vision=True),)),
        ProviderStatus(name="ollama", url="http://ollama.invalid", configured=True, available=True,
                       models=(_model("ollama:qwen3:8b", "ollama", 8.2),)),
        ProviderStatus(name="ds4", configured=False),
    ))


def _settings(**kw):
    base = dict(source_root=Path("/tmp/src"), archive_root=Path("/tmp/arc"))
    base.update(kw)
    return Settings(**base)


def test_provider_urls_are_carried_into_the_analysis_config():
    urls = {"ollama": "http://ollama.invalid", "vllm": "http://vllm.invalid", "ds4": ""}
    cfg = build_analysis_config(settings=_settings(providers=urls), discovery=_discovery(), taxonomy=_TAXONOMY)
    assert cfg.provider_urls == urls


def test_auto_selection_prefers_vllm_for_every_role():
    cfg = build_analysis_config(settings=_settings(), discovery=_discovery(), taxonomy=_TAXONOMY)
    assert cfg.text_models[0] == "vllm:qwen3.6-27b"
    assert cfg.vision_models[0] == "vllm:qwen3.6-27b"


def test_a_pinned_model_goes_first_without_dropping_the_others():
    cfg = build_analysis_config(
        settings=_settings(facts_model="ollama:qwen3:8b"),
        discovery=_discovery(),
        taxonomy=_TAXONOMY,
    )
    assert cfg.text_models[0] == "ollama:qwen3:8b"
    assert "vllm:qwen3.6-27b" in cfg.text_models


def test_no_discovery_yields_empty_candidate_lists():
    cfg = build_analysis_config(settings=_settings(), discovery=None, taxonomy=_TAXONOMY)
    assert cfg.text_models == ()
    assert cfg.vision_models == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_task_builders.py -v`
Expected: FAIL con `TypeError: Settings.__init__() got an unexpected keyword argument 'providers'` oppure `AttributeError: 'AnalysisConfig' object has no attribute 'provider_urls'` se la Task 9 non è stata completata

- [ ] **Step 3: Write minimal implementation**

Sostituisci integralmente `archiver/task_builders.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from .analyzer import AnalysisConfig
from .model_selection import ROLE_CLASSIFY, ROLE_FACTS, ROLE_VISION, rank_models

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import DiscoveryResult
    from .settings import Settings
    from .taxonomy import Taxonomy


def _with_pin(candidates: tuple[str, ...], pinned: str) -> tuple[str, ...]:
    if not pinned or pinned == "auto":
        return candidates
    return (pinned, *tuple(c for c in candidates if c != pinned))


def build_analysis_config(
    *, settings: "Settings", discovery: "DiscoveryResult | None", taxonomy: "Taxonomy"
) -> AnalysisConfig:
    models = discovery.models if discovery else ()
    text_models = _with_pin(rank_models(models, ROLE_FACTS), settings.facts_model)
    vision_models = _with_pin(rank_models(models, ROLE_VISION), settings.vision_model)
    return AnalysisConfig(
        output_language=settings.output_language,
        taxonomy=taxonomy,
        text_models=text_models,
        vision_models=vision_models,
        filename_separator=settings.filename_separator,
        ocr_mode=settings.ocr_mode,
        provider_urls=dict(settings.providers),
    )
```

In `archiver/app.py`:

- elimina il metodo `_ordered_classify_models` (`:141-156`) e sostituisci il suo unico call site (intorno a `:554`) con `rank_models(self._discovery.models, ROLE_CLASSIFY)` più il pin di `settings.classify_model` tramite la stessa logica di `_with_pin` (importala da `task_builders`)
- `_run_discovery` (`:400`) passa `self.settings.providers` e la probe cache: `discover_providers(self.settings.providers, probe_cache=load_probe_cache())`
- `action_settings` (`:316`) passa `providers=self.settings.providers` e `available_models=self._discovery.models if self._discovery else ()`
- `_on_settings_done` (`:359`) confronta `result.providers != self.settings.providers` per decidere se rilanciare la discovery, e non passa più `vision_model_fallback`
- ogni `base_url=self.settings.ollama_base_url, ds4_base_url=self.settings.ds4_base_url` diventa `provider_urls=self.settings.providers`
- aggiungi il binding del doctor, che la Task 14 collegherà: `Binding("d", "doctor", "Doctor", show=True)`

In `archiver/ui_status.py`, riscrivi `provider_summary`:

```python
def provider_summary(discovery: "DiscoveryResult | None", settings: "Settings") -> str:
    if not discovery:
        return ""
    from .model_selection import ROLE_CLASSIFY, ROLE_FACTS, ROLE_VISION, rank_models

    names = [s.name if s.available else f"{s.name}(down)"
             for s in discovery.providers if s.configured]
    if not names:
        return ""
    models = discovery.models

    def pick(pinned: str, role: str) -> str:
        if pinned and pinned != "auto":
            return pinned
        ranked = rank_models(models, role)
        return ranked[0] if ranked else "none"

    count = f"{len(models)} models" if models else "no models"
    return (
        f"{'+'.join(names)} • {count}"
        f" • facts={pick(settings.facts_model, ROLE_FACTS)}"
        f" • classify={pick(settings.classify_model, ROLE_CLASSIFY)}"
        f" • vision={pick(settings.vision_model, ROLE_VISION)}"
    )
```

Aggiorna il call site in `app.py:412` togliendo l'argomento `model_picker`.

In `archiver/__main__.py`, sostituisci nel costruttore di `Settings` i due campi piatti e il fallback vision con `providers=cfg.providers`.

- [ ] **Step 4: Run test to verify it passes**

```bash
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v
~/.local/share/amenity-stuff/venv/bin/python -c "import archiver.app"
TOKENSAVE_DISABLE_GREP_HOOK=1 grep -rn "ds4_base_url\|ollama_base_url\|vision_model_fallback\|pick_model_candidates" archiver/
```
Expected: pytest PASS; l'import non solleva; il grep non restituisce nulla

- [ ] **Step 5: Commit**

```bash
git add archiver/task_builders.py archiver/ui_status.py archiver/app.py archiver/__main__.py tests/test_task_builders.py
git commit -m "refactor: wire the app to the provider registry and unified ranking"
```

---

### Task 12: Diagnosi

Modulo puro. Il probe è iniettato come callable, quindi i test non toccano la rete.

**Files:**
- Create: `archiver/doctor.py`, `archiver/model_catalog.py`
- Test: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `discovery.DiscoveryResult`, `model_selection.rank_models`, `capabilities.SOURCE_*`, `providers.PROVIDERS`
- Produces:
  - `Remedy` (frozen): `kind: str`, `model: str`, `provider: str`, `size_bytes: int`, `note: str`
  - `Check` (frozen): `key: str`, `label: str`, `status: str`, `detail: str`, `remedies: tuple[Remedy, ...] = ()`
  - `DoctorReport` (frozen): `checks: tuple[Check, ...]`, proprietà `worst -> str`, `exit_code -> int`
  - `run_doctor(*, discovery: DiscoveryResult, settings, probe=None) -> DoctorReport`
  - `model_catalog.catalog_for_role(role: str) -> tuple[CatalogEntry, ...]` con `CatalogEntry(tag: str, size_bytes: int, note: str)`

- [ ] **Step 1: Write the failing test**

Crea `tests/test_doctor.py`:

```python
from __future__ import annotations

from pathlib import Path

from archiver.capabilities import (
    CAP_COMPLETION,
    CAP_VISION,
    SOURCE_DECLARED,
    SOURCE_HEURISTIC,
)
from archiver.discovery import DiscoveryResult, ModelInfo, ProviderStatus
from archiver.doctor import run_doctor
from archiver.settings import Settings


def _settings(**kw):
    base = dict(source_root=Path("/tmp/src"), archive_root=Path("/tmp/arc"))
    base.update(kw)
    return Settings(**base)


def _model(model_id, provider, *, vision=False, source=SOURCE_DECLARED, size=8.0):
    caps = {CAP_COMPLETION} | ({CAP_VISION} if vision else set())
    return ModelInfo(id=model_id, provider=provider, capabilities=frozenset(caps),
                     parameter_size_b=size, capability_source=source)


def _check(report, key):
    return next(c for c in report.checks if c.key == key)


def test_report_has_one_check_per_provider_plus_two_roles():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", configured=False),
        ProviderStatus(name="ollama", configured=False),
        ProviderStatus(name="ds4", configured=False),
    )), settings=_settings())
    keys = [c.key for c in report.checks]
    assert keys == ["provider.vllm", "provider.ollama", "provider.ds4",
                    "role.text", "role.vision"]


def test_unconfigured_provider_is_skipped_not_failed():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", configured=False),
    )), settings=_settings())
    assert _check(report, "provider.vllm").status == "skip"


def test_unreachable_provider_fails_with_the_real_reason():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://x.invalid", configured=True,
                       available=False, detail="ConnectionRefusedError: nope"),
    )), settings=_settings())
    check = _check(report, "provider.vllm")
    assert check.status == "fail"
    assert "ConnectionRefusedError" in check.detail


def test_reachable_provider_without_models_warns():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="ollama", url="http://x.invalid", configured=True,
                       available=True, models=()),
    )), settings=_settings())
    assert _check(report, "provider.ollama").status == "warn"


def test_missing_vision_fails_and_offers_a_pull_when_ollama_is_reachable():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="ollama", url="http://x.invalid", configured=True,
                       available=True, models=(_model("ollama:qwen3:8b", "ollama"),)),
    )), settings=_settings())
    vision = _check(report, "role.vision")
    assert vision.status == "fail"
    assert vision.remedies
    assert all(r.kind == "pull" and r.provider == "ollama" for r in vision.remedies)


def test_missing_vision_offers_only_hints_when_no_installable_provider_is_up():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://x.invalid", configured=True,
                       available=True, models=(_model("vllm:qwen3.6-27b", "vllm"),)),
        ProviderStatus(name="ollama", configured=False),
    )), settings=_settings())
    vision = _check(report, "role.vision")
    assert vision.status == "fail"
    assert all(r.kind == "hint" for r in vision.remedies)


def test_heuristic_only_vision_warns_instead_of_passing():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://x.invalid", configured=True, available=True,
                       models=(_model("vllm:llava-ish", "vllm", vision=True,
                                      source=SOURCE_HEURISTIC),)),
    )), settings=_settings())
    assert _check(report, "role.vision").status == "warn"


def test_probe_promotes_a_heuristic_vision_model_to_ok():
    model = _model("vllm:qwen3.6-27b", "vllm", source=SOURCE_HEURISTIC)
    discovery = DiscoveryResult(providers=(
        ProviderStatus(name="vllm", url="http://x.invalid", configured=True,
                       available=True, models=(model,)),
    ))

    def probe(*, url, bare_id):
        return True  # il modello accetta immagini

    report = run_doctor(discovery=discovery, settings=_settings(), probe=probe)
    assert _check(report, "role.vision").status == "ok"


def test_probe_is_not_called_for_declared_capabilities():
    calls = []

    def probe(*, url, bare_id):
        calls.append(bare_id)
        return True

    run_doctor(
        discovery=DiscoveryResult(providers=(
            ProviderStatus(name="ollama", url="http://x.invalid", configured=True, available=True,
                           models=(_model("ollama:llava:7b", "ollama", vision=True),)),
        )),
        settings=_settings(),
        probe=probe,
    )
    assert calls == []


def test_pinned_model_that_disappeared_warns():
    report = run_doctor(
        discovery=DiscoveryResult(providers=(
            ProviderStatus(name="ollama", url="http://x.invalid", configured=True, available=True,
                           models=(_model("ollama:qwen3:8b", "ollama"),)),
        )),
        settings=_settings(facts_model="ollama:disappeared"),
    )
    text = _check(report, "role.text")
    assert text.status == "warn"
    assert "disappeared" in text.detail


def test_worst_and_exit_code_reflect_the_severity():
    report = run_doctor(discovery=DiscoveryResult(providers=(
        ProviderStatus(name="ollama", configured=False),
    )), settings=_settings())
    assert report.worst == "fail"   # nessun modello di testo
    assert report.exit_code == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_doctor.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'archiver.doctor'`

- [ ] **Step 3: Write minimal implementation**

Crea `archiver/model_catalog.py`:

```python
"""Rosa curata dei modelli installabili, per ruolo.

Stesso criterio di manutenzione di CURATED_BIAS: a mano, ogni tanto, con
task dedicate. Le dimensioni sono indicative e servono a far scegliere
consapevolmente prima di scaricare gigabyte su una macchina remota.
"""
from __future__ import annotations

from dataclasses import dataclass

_GB = 1024 ** 3


@dataclass(frozen=True)
class CatalogEntry:
    tag: str
    size_bytes: int
    note: str


_VISION = (
    CatalogEntry("moondream:latest", int(1.7 * _GB), "leggero e veloce"),
    CatalogEntry("llava:7b", int(4.7 * _GB), "qualità migliore"),
    CatalogEntry("minicpm-v:latest", int(5.5 * _GB), "ottimo sul testo nelle immagini"),
)

_TEXT = (
    CatalogEntry("qwen2.5:3b-instruct", int(1.9 * _GB), "piccolo e affidabile"),
    CatalogEntry("qwen3:8b", int(5.2 * _GB), "buon compromesso per la classificazione"),
)


def catalog_for_role(role: str) -> tuple[CatalogEntry, ...]:
    return _VISION if role == "vision" else _TEXT
```

Crea `archiver/doctor.py`:

```python
"""Diagnosi dei provider e dei modelli.

Modulo puro: riceve il risultato della scoperta e restituisce un report.
L'unico I/O possibile è il probe, che arriva iniettato come callable così
i test non toccano la rete e la UI decide se pagarne il costo.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, TYPE_CHECKING

from .capabilities import CAP_VISION, SOURCE_HEURISTIC, SOURCE_PROBED
from .model_catalog import catalog_for_role
from .model_selection import ROLE_FACTS, ROLE_VISION, rank_models
from .providers import KIND_OLLAMA, PROVIDERS, split_model_id

if TYPE_CHECKING:  # pragma: no cover
    from .discovery import DiscoveryResult, ModelInfo
    from .settings import Settings

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"

_SEVERITY = {STATUS_SKIP: 0, STATUS_OK: 1, STATUS_WARN: 2, STATUS_FAIL: 3}


@dataclass(frozen=True)
class Remedy:
    kind: str          # "pull" | "hint"
    model: str
    provider: str
    size_bytes: int = 0
    note: str = ""


@dataclass(frozen=True)
class Check:
    key: str
    label: str
    status: str
    detail: str = ""
    remedies: tuple[Remedy, ...] = ()


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[Check, ...] = ()

    @property
    def worst(self) -> str:
        if not self.checks:
            return STATUS_SKIP
        return max((c.status for c in self.checks), key=lambda s: _SEVERITY.get(s, 0))

    @property
    def exit_code(self) -> int:
        return 1 if self.worst == STATUS_FAIL else 0


def _provider_check(status) -> Check:
    label = f"{status.name} {status.url}".strip()
    if not status.configured:
        return Check(f"provider.{status.name}", label, STATUS_SKIP, "non configurato")
    if not status.available:
        return Check(f"provider.{status.name}", label, STATUS_FAIL, status.detail)
    if not status.models:
        return Check(f"provider.{status.name}", label, STATUS_WARN,
                     "raggiungibile, nessun modello")
    return Check(f"provider.{status.name}", label, STATUS_OK,
                 f"{len(status.models)} modelli")


def _installable_provider(discovery) -> Optional[str]:
    for spec in PROVIDERS:
        if not spec.supports_install:
            continue
        status = discovery.status(spec.name)
        if status is not None and status.available:
            return spec.name
    return None


def _remedies(discovery, role: str) -> tuple[Remedy, ...]:
    target = _installable_provider(discovery)
    entries = catalog_for_role(role)
    if target:
        return tuple(
            Remedy(kind="pull", model=e.tag, provider=target,
                   size_bytes=e.size_bytes, note=e.note)
            for e in entries
        )
    return tuple(
        Remedy(kind="hint", model=e.tag, provider="", size_bytes=e.size_bytes,
               note="avvia il server con questo modello, oppure configura ollama")
        for e in entries
    )


def _apply_probe(discovery: "DiscoveryResult", probe) -> tuple["ModelInfo", ...]:
    models = discovery.models
    if probe is None:
        return models
    out = []
    for model in models:
        if model.capability_source != SOURCE_HEURISTIC:
            out.append(model)
            continue
        spec, bare = split_model_id(model.id)
        if spec.kind == KIND_OLLAMA:
            out.append(model)
            continue
        status = discovery.status(model.provider)
        if status is None or not status.url:
            out.append(model)
            continue
        try:
            verdict = probe(url=status.url, bare_id=bare)
        except Exception:
            verdict = None
        if verdict is True:
            caps = frozenset(model.capabilities | {CAP_VISION})
            out.append(replace(model, capabilities=caps, capability_source=SOURCE_PROBED))
        elif verdict is False:
            caps = frozenset(model.capabilities - {CAP_VISION})
            out.append(replace(model, capabilities=caps, capability_source=SOURCE_PROBED))
        else:
            out.append(model)
    return tuple(out)


def _role_check(models, *, key: str, label: str, role: str, discovery, pinned: str) -> Check:
    ranked = rank_models(models, role)
    if not ranked:
        return Check(key, label, STATUS_FAIL, "nessun modello disponibile",
                     _remedies(discovery, role))
    if pinned and pinned != "auto" and pinned not in ranked:
        return Check(key, label, STATUS_WARN,
                     f"modello fissato non trovato: {pinned} (userei {ranked[0]})")
    chosen = pinned if (pinned and pinned in ranked) else ranked[0]
    by_id = {m.id: m for m in models}
    if by_id[chosen].capability_source == SOURCE_HEURISTIC:
        return Check(key, label, STATUS_WARN,
                     f"{chosen} (capability dedotta dal nome, non confermata)")
    return Check(key, label, STATUS_OK, chosen)


def run_doctor(
    *,
    discovery: "DiscoveryResult",
    settings: "Settings",
    probe: Optional[Callable[..., Optional[bool]]] = None,
) -> DoctorReport:
    checks = [_provider_check(s) for s in discovery.providers]
    models = _apply_probe(discovery, probe)
    checks.append(_role_check(models, key="role.text", label="modello semantico",
                              role=ROLE_FACTS, discovery=discovery,
                              pinned=settings.facts_model))
    checks.append(_role_check(models, key="role.vision", label="modello vision",
                              role=ROLE_VISION, discovery=discovery,
                              pinned=settings.vision_model))
    return DoctorReport(checks=tuple(checks))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_doctor.py -v`
Expected: PASS, 11 test

- [ ] **Step 5: Commit**

```bash
git add archiver/doctor.py archiver/model_catalog.py tests/test_doctor.py
git commit -m "feat: add provider and model diagnosis with actionable remedies"
```

---

### Task 13: Download dei modelli su Ollama

**Files:**
- Create: `archiver/ollama_admin.py`
- Test: `tests/test_ollama_admin.py`

**Interfaces:**
- Consumes: niente dal progetto
- Produces:
  - `PullProgress` (frozen): `status: str`, `completed: int`, `total: int`, proprietà `fraction -> float`
  - `pull_model(*, base_url: str, model: str, on_progress=None, should_cancel=None, opener=None, timeout_s: float = 3600.0) -> str | None` → `None` se riuscito, altrimenti il messaggio d'errore
  - `probe_vision(*, base_url: str, model: str, opener=None, timeout_s: float = 30.0) -> bool | None`

- [ ] **Step 1: Write the failing test**

Crea `tests/test_ollama_admin.py`:

```python
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
    assert error == "annullato"


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_ollama_admin.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'archiver.ollama_admin'`

- [ ] **Step 3: Write minimal implementation**

Crea `archiver/ollama_admin.py`:

```python
"""Operazioni amministrative: download dei modelli e probe delle capability.

Il download avviene SULLA MACCHINA CHE OSPITA OLLAMA, non su quella che
esegue amenity-ai: puntando a un host remoto i gigabyte finiscono lì.

La cancellazione è cooperativa e segue la convenzione già usata da scan,
classify e move: un callback should_cancel interrogato fra un blocco e
l'altro dello stream.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.request import Request, urlopen

from .capabilities import interpret_probe

_PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)


@dataclass(frozen=True)
class PullProgress:
    status: str = ""
    completed: int = 0
    total: int = 0

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.completed / self.total)


def pull_model(
    *,
    base_url: str,
    model: str,
    on_progress: Optional[Callable[[PullProgress], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    opener: Optional[Callable[..., Any]] = None,
    timeout_s: float = 3600.0,
) -> Optional[str]:
    """Scarica un modello. Ritorna None se riuscito, altrimenti l'errore."""
    send = opener or urlopen
    request = Request(
        base_url.rstrip("/") + "/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with send(request, timeout=timeout_s) as resp:
            for raw in resp:
                if should_cancel is not None and should_cancel():
                    return "annullato"
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                error = data.get("error")
                if error:
                    return str(error)
                if on_progress is not None:
                    on_progress(
                        PullProgress(
                            status=str(data.get("status") or ""),
                            completed=int(data.get("completed") or 0),
                            total=int(data.get("total") or 0),
                        )
                    )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def probe_vision(
    *,
    base_url: str,
    model: str,
    opener: Optional[Callable[..., Any]] = None,
    timeout_s: float = 30.0,
) -> Optional[bool]:
    """Chiede al modello di guardare un PNG 1x1. None = non conclusivo."""
    send = opener or urlopen
    payload = {
        "model": model,
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ok?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PNG_1X1}"},
                    },
                ],
            }
        ],
    }
    request = Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with send(request, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200) or 200
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        status = getattr(exc, "code", 0) or 0
        try:
            body = exc.read().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            body = str(exc)
    return interpret_probe(status=status, body=body)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/test_ollama_admin.py -v`
Expected: PASS, 7 test

- [ ] **Step 5: Commit**

```bash
git add archiver/ollama_admin.py tests/test_ollama_admin.py
git commit -m "feat: pull ollama models with progress and cooperative cancellation"
```

---

### Task 14: Schermata doctor e sottocomando CLI

Ultimo strato: UI e punto d'ingresso. Nessun test automatico (il progetto non testa la TUI); la verifica è manuale sotto PTY.

**Files:**
- Create: `archiver/doctor_screen.py`
- Modify: `archiver/app.py` (azione `action_doctor`), `archiver/__main__.py` (sottocomando), `archiver/help_screen.py` (documenta `[d]`)

**Interfaces:**
- Consumes: `doctor.run_doctor`, `doctor.DoctorReport`, `doctor.Remedy`; `ollama_admin.pull_model`, `ollama_admin.probe_vision`; `probe_cache.save_probe_result`; `discovery.discover_providers`
- Produces: `DoctorScreen(settings, discovery, on_refresh)` — `ModalScreen[None]`

- [ ] **Step 1: Write the screen**

Crea `archiver/doctor_screen.py`:

```python
"""Schermata del doctor: mostra il report e sa rimediare.

Nessuna logica di diagnosi qui dentro: run_doctor decide, questa classe
rende. Il probe e il download girano in worker perché non devono mai
bloccare l'event loop di Textual.
"""
from __future__ import annotations

from typing import Callable, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, OptionList, Static

from .capabilities import CAP_COMPLETION, CAP_VISION
from .discovery import discover_providers
from .doctor import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIP,
    STATUS_WARN,
    DoctorReport,
    Remedy,
    run_doctor,
)
from .ollama_admin import PullProgress, probe_vision, pull_model
from .probe_cache import load_probe_cache, save_probe_result

_ICON = {STATUS_OK: "✓", STATUS_WARN: "⚠", STATUS_FAIL: "✗", STATUS_SKIP: "—"}
_GB = 1024 ** 3


def _network_probe(*, url: str, bare_id: str) -> Optional[bool]:
    """Interroga il modello e memorizza solo gli esiti conclusivi."""
    verdict = probe_vision(base_url=url, model=bare_id)
    if verdict is not None:
        caps = {CAP_COMPLETION} | ({CAP_VISION} if verdict else set())
        save_probe_result(url=url, bare_id=bare_id, capabilities=frozenset(caps))
    return verdict


class DoctorScreen(ModalScreen[None]):
    CSS = """
    DoctorScreen { layout: vertical; }
    #intro { height: auto; color: $text-muted; }
    #report { height: auto; border: round $accent; background: $panel; padding: 1 2; }
    #remedies { height: auto; border: round $accent; background: $panel; }
    #pull_status { height: auto; padding: 1 2; }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
        Binding("r", "refresh", "Refresh"),
        Binding("x", "cancel_pull", "Cancel download"),
    ]

    def __init__(
        self,
        *,
        settings,
        discovery=None,
        on_refresh: Optional[Callable[[], None]] = None,
        on_report: Optional[Callable[[DoctorReport], None]] = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._discovery = discovery
        self._on_refresh = on_refresh
        self._on_report = on_report
        self._report: Optional[DoctorReport] = None
        self._remedies: tuple[Remedy, ...] = ()
        self._cancel_pull = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "Doctor: r ricontrolla • Enter installa • x annulla • Esc chiude",
            id="intro",
        )
        yield Static("Analisi in corso…", id="report", markup=False)
        yield OptionList(id="remedies")
        yield Static("", id="pull_status", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh()

    # --- diagnosi -------------------------------------------------------

    def action_refresh(self) -> None:
        self.query_one("#pull_status", Static).update("")
        self.run_worker(self._diagnose, thread=True, exclusive=True)

    def _diagnose(self) -> None:
        discovery = discover_providers(
            self._settings.providers, probe_cache=load_probe_cache()
        )
        report = run_doctor(
            discovery=discovery, settings=self._settings, probe=_network_probe
        )
        self.app.call_from_thread(self._render, discovery, report)

    def _render(self, discovery, report: DoctorReport) -> None:
        self._discovery = discovery
        self._report = report
        if self._on_report is not None:
            self._on_report(report)

        lines = [
            f"{_ICON.get(check.status, '?')} {check.label} — {check.detail}"
            for check in report.checks
        ]
        self.query_one("#report", Static).update("\n".join(lines))

        self._remedies = tuple(r for check in report.checks for r in check.remedies)
        option_list = self.query_one("#remedies", OptionList)
        option_list.clear_options()
        for remedy in self._remedies:
            gb = remedy.size_bytes / _GB
            if remedy.kind == "pull":
                option_list.add_option(
                    f"installa {remedy.model} su {remedy.provider} — {gb:.1f} GB · {remedy.note}"
                )
            else:
                option_list.add_option(f"(manuale) {remedy.model} — {remedy.note}")

    # --- installazione --------------------------------------------------

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "remedies":
            return
        remedy = self._remedies[event.option_index]
        if remedy.kind != "pull":
            return  # gli hint non sono azionabili: non c'è API per farlo
        status = self._discovery.status(remedy.provider) if self._discovery else None
        if status is None or not status.url:
            return

        self._cancel_pull = False
        gb = remedy.size_bytes / _GB
        # Il download avviene sulla macchina che ospita Ollama, non su questa:
        # dirlo esplicitamente prima di partire è parte della conferma.
        self.query_one("#pull_status", Static).update(
            f"Scarico {remedy.model} su {status.url} (~{gb:.1f} GB). x per annullare."
        )
        url, model = status.url, remedy.model
        self.run_worker(lambda: self._pull(url, model), thread=True, exclusive=True)

    def _pull(self, url: str, model: str) -> None:
        def on_progress(progress: PullProgress) -> None:
            if progress.total:
                text = (
                    f"{progress.status} "
                    f"{progress.completed / _GB:.1f}/{progress.total / _GB:.1f} GB"
                )
            else:
                text = progress.status
            self.app.call_from_thread(self._set_pull_status, text)

        error = pull_model(
            base_url=url,
            model=model,
            on_progress=on_progress,
            should_cancel=lambda: self._cancel_pull,
        )
        if error:
            self.app.call_from_thread(self._set_pull_status, f"✗ {error}")
            return
        self.app.call_from_thread(self._set_pull_status, "✓ installato")
        self.app.call_from_thread(self.action_refresh)
        if self._on_refresh is not None:
            self.app.call_from_thread(self._on_refresh)

    def _set_pull_status(self, text: str) -> None:
        self.query_one("#pull_status", Static).update(text)

    def action_cancel_pull(self) -> None:
        self._cancel_pull = True

    def action_close(self) -> None:
        self.dismiss(None)


class _DoctorApp(App):
    """App minimale per il sottocomando CLI: solo la schermata doctor."""

    def __init__(self, settings) -> None:
        super().__init__()
        self._settings = settings
        self.report: Optional[DoctorReport] = None

    def on_mount(self) -> None:
        self.push_screen(
            DoctorScreen(settings=self._settings, on_report=self._remember),
            callback=lambda _: self.exit(),
        )

    def _remember(self, report: DoctorReport) -> None:
        self.report = report


def run_doctor_cli(settings) -> int:
    """Esegue il doctor da solo e ritorna l'exit code del report."""
    app = _DoctorApp(settings)
    app.run(mouse=False)
    return app.report.exit_code if app.report is not None else 1
```

- [ ] **Step 2: Wire the two surfaces**

In `archiver/app.py`, aggiungi `from .doctor_screen import DoctorScreen` agli import e implementa l'azione già associata al binding aggiunto nella Task 11:

```python
    async def action_doctor(self) -> None:
        if self._analysis_task.running or self._archive_task.running or self._scan_task.running:
            return
        self.push_screen(
            DoctorScreen(
                settings=self.settings,
                discovery=self._discovery,
                on_refresh=lambda: self.run_worker(self._run_discovery()),
            ),
            wait_for_dismiss=False,
        )
```

In `archiver/__main__.py`, aggiungi il sottocomando accanto a `run` e `report`:

```python
    sub.add_parser("doctor", help="Verifica provider e modelli disponibili")
```

e, nel corpo di `main()`, prima del ramo che avvia la TUI:

```python
    if getattr(args, "command", None) == "doctor":
        cfg = load_config()
        settings = Settings(
            source_root=args.source,
            archive_root=args.archive,
            providers=cfg.providers,
            facts_model=cfg.facts_model,
            classify_model=cfg.classify_model,
            vision_model=cfg.vision_model,
        )
        raise SystemExit(run_doctor_cli(settings))
```

con `from .doctor_screen import run_doctor_cli` fra gli import di `__main__.py`.

In `archiver/help_screen.py`, aggiungi la riga per `d` — "Doctor: verifica provider e modelli".

- [ ] **Step 3: Verify manually under a PTY**

L'utente lancia sempre da un terminale interattivo: verificare con output pipeato non riprodurrebbe le stesse code-path.

```bash
script -qec "amenity-ai doctor" /dev/null
echo "exit code: $?"
```

Attese, con l'assetto attuale (Ollama e vLLM su, ds4 vuoto):
- `ds4` compare come `skip` "non configurato"
- `role.vision` è `ok` **dopo** il probe su vLLM, e alla riapertura successiva è `ok` già dal percorso veloce grazie alla probe cache
- con Ollama spento, `provider.ollama` è `fail` con il tipo d'eccezione vero
- l'exit code è `0` quando nessun check è `fail`

Poi la TUI:

```bash
script -qec "amenity-ai --source /tmp --archive /tmp/ARCHIVE" /dev/null
```
Premere `d`, verificare che la schermata si apra, che `escape` la chiuda, e che nessuna apertura sia automatica all'avvio.

- [ ] **Step 4: Commit**

```bash
git add archiver/doctor_screen.py archiver/app.py archiver/__main__.py archiver/help_screen.py
git commit -m "tui: add doctor screen and amenity-ai doctor subcommand"
```

---

### Task 15: Documentazione e rilascio 0.12.0

**Files:**
- Modify: `VERSION`, `pyproject.toml`, `CHANGELOG.md`, `CLAUDE.md`, `PROJECT_SPEC.md`, `README.md`

- [ ] **Step 1: Run the full suite and a smoke test**

```bash
~/.local/share/amenity-stuff/venv/bin/python -m pytest tests/ -v
~/.local/share/amenity-stuff/venv/bin/pip install -e .
script -qec "amenity-ai doctor" /dev/null
```
Expected: tutti i test passano; il doctor stampa il report e l'exit code è coerente

- [ ] **Step 2: Bump the version**

`VERSION` diventa `0.12.0`; in `pyproject.toml` la stessa stringa. Verifica:

```bash
cat VERSION
TOKENSAVE_DISABLE_GREP_HOOK=1 grep -n "version" pyproject.toml
```
I due valori devono coincidere.

- [ ] **Step 3: Update the docs**

- `CHANGELOG.md`: nuova sezione `0.12.0` con supporto vLLM, configurazione a tre URL con scoperta automatica, `amenity-ai doctor` con installazione modelli, ranking basato sui metadati, prefissi espliciti, e la nota che `vision_model_fallback` è stata rimossa e le config vengono migrate automaticamente
- `CLAUDE.md`: riscrivi le sezioni *Model Selection ("auto")* e *Two Configuration Dataclasses*, che descrivono l'assetto a due provider. Documenta il registry, i tre `kind`, la priorità `vllm > ollama > ds4`, il ranking a quattro criteri e i moduli nuovi nella sezione *Module Organization*
- `PROJECT_SPEC.md` e `README.md`: aggiorna le istruzioni di configurazione e aggiungi `amenity-ai doctor` ai comandi

Vincolo: nessun hostname reale in nessuno di questi file.

- [ ] **Step 4: Verify nothing leaked**

Il vincolo "nessun hostname reale nel repo" vale per **ogni** file versionato, `docs/` inclusi. I test devono usare solo domini riservati (`.invalid`, come già fanno) e i default restano `http://localhost:11434` e stringa vuota.

```bash
TOKENSAVE_DISABLE_GREP_HOOK=1 grep -rniE "\.local\b|[0-9]{1,3}(\.[0-9]{1,3}){3}" \
  --exclude-dir=.git --exclude-dir=.tokensave . \
  | grep -v "localhost" | grep -v "127.0.0.1" || echo "nessun hostname reale nel repo"
```
Expected: nessun risultato oltre a `localhost`. Se qualcosa compare, va rimosso prima del commit.

- [ ] **Step 5: Commit**

```bash
git add VERSION pyproject.toml CHANGELOG.md CLAUDE.md PROJECT_SPEC.md README.md
git commit -m "docs: document the three-provider redesign and release 0.12.0"
```

---

## Note per chi esegue

- **Ordine obbligato.** Le Task 1-3 sono indipendenti fra loro ma tutto il resto ne dipende. Le Task 4-8 vanno in sequenza. La Task 11 richiede 9 e 10 completate, altrimenti `app.py` non compila.
- **Suite verde a ogni commit.** Le task che cambiano un contratto riscrivono i test di quel contratto nella task stessa.
- **Il grep di controllo** `TOKENSAVE_DISABLE_GREP_HOOK=1 grep -rn "ds4_base_url\|ollama_base_url\|vision_model_fallback\|pick_model_candidates" archiver/` deve restituire zero risultati dopo la Task 11.
- **Verifica finale sotto PTY**, mai con output pipeato: l'utente lancia sempre da zsh interattivo e diversi comportamenti (progress bar, prompt, `isatty()`) differiscono.
