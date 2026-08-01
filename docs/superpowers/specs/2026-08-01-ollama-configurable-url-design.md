# Design: URL di Ollama configurabile (server remoto)

Data: 2026-08-01
Stato: approvato (design)

## Contesto

Con l'introduzione del provider ds4 (spec 2026-08-01-ds4-openai-backend-design.md),
Ollama è rimasto l'unico provider con endpoint fisso: `AnalysisConfig.ollama_base_url`
esiste ma nessuno lo imposta (resta il default `http://localhost:11434`), `app.py`
hardcoda l'URL nei due call site classify, e la discovery usa il CLI `ollama list` —
che su una macchina remota non esiste. Ollama può girare su un altro host della rete
locale: l'URL va reso configurabile, simmetrico a `ds4_base_url`.

## Decisioni

1. **Default `http://localhost:11434`** (mai vuoto): Ollama resta il provider base; il
   default riproduce il comportamento attuale. Nessun impatto per chi non configura nulla.
2. **Discovery via HTTP**: `GET {url}/api/tags` sostituisce il CLI `ollama list`. Il CLI
   parla comunque col server locale, quindi il passaggio a HTTP non perde alcun caso
   reale e abilita il caso remoto. Il check "binario in PATH" sparisce.
3. **Privacy invariata**: nessun hostname personale nel repo; gli esempi usano
   `http://localhost:11434`. Il guard `scripts/check_no_private_host.sh` viene esteso a
   controllare anche l'hostname di `ollama_base_url` dalla config locale.

## Non-obiettivi

- Nessuna autenticazione (Ollama non la supporta nativamente).
- Nessun supporto multi-istanza Ollama (un solo endpoint).
- Nessun cambiamento a modelli, ranking, fallback: i meccanismi per-candidato appena
  introdotti coprono già "Ollama remoto giù a metà sessione".

## Modifiche

### Config e threading (pattern identico a ds4_base_url)

- `config.AppConfig.ollama_base_url: str = "http://localhost:11434"` — persistito;
  parsing difensivo in `load_config()` (stringa non vuota → strip; altrimenti default).
- `settings.Settings.ollama_base_url: str = "http://localhost:11434"`.
- `setup_logic.py`: entrambi i convertitori (`settings_from_setup`,
  `app_config_from_settings`) propagano il campo.
- `__main__.py`: `Settings(..., ollama_base_url=cfg.ollama_base_url)`.
- `task_builders.build_analysis_config()`: `AnalysisConfig(...,
  ollama_base_url=settings.ollama_base_url)` (il campo esiste già in `AnalysisConfig`).
- `app.py`: nei due call site classify (`_run_classify_batch`, `_run_classify_row`)
  l'hardcoded `base_url="http://localhost:11434"` diventa
  `base_url=self.settings.ollama_base_url`.
- Il path vision (`extract_image_smart`) legge già `config.ollama_base_url`: eredita il
  valore senza modifiche.

### Discovery (`discovery.py`)

- `_discover_ollama(base_url: str)` → `GET {base_url}/api/tags` con timeout 2.5 s via
  `_get_json` (già esistente). Payload `{"models": [{"name": "..."}]}` → nomi modello;
  lista vuota → `details="OK (no models listed)"`; errore/timeout →
  `ProviderInfo(name="ollama", available=False, details="Not reachable (<tipo errore>)")`.
- `discover_providers(*, ollama_base_url: str = "http://localhost:11434",
  ds4_base_url: str = "")`.
- `shutil.which`/`subprocess`/`_run` non servono più: rimossi se non usati altrove.
- `app.py` `do_discover()` passa `ollama_base_url=self.settings.ollama_base_url`.

### Settings screen (`settings_screen.py`)

- `SettingsResult.ollama_base_url: str`; parametro keyword nel costruttore.
- Secondo `Input` (id `ollama_url`, placeholder `http://localhost:11434`) sotto il campo
  ds4, con etichetta "ollama endpoint:". Helper `_current_ollama_url()` con fallback
  pre-compose, letto in entrambi i dismiss path (stessa meccanica di `_current_ds4_url`).
- `app.py` `action_settings` passa `ollama_base_url=self.settings.ollama_base_url`;
  `_on_settings_done` applica il campo nel `replace(...)` e ri-esegue la discovery se
  ds4 **o** ollama URL sono cambiati.

### UI (`ui_status.py`)

- `provider_summary`: l'etichetta `ollama(missing)` diventa `ollama(down)` — con la
  discovery HTTP indica "server non raggiungibile", non "binario assente".

### Guard privacy (`scripts/check_no_private_host.sh`)

- Legge dalla config locale gli hostname sia di `ds4_base_url` sia di `ollama_base_url`;
  per ciascuno (se non vuoto/localhost/127.0.0.1) fallisce se compare nei file tracciati.

## Error handling

Identico a oggi: server irraggiungibile in discovery → provider non disponibile, nessun
candidato; errore a runtime → fallback per-candidato esistente. Nessuna eccezione verso
il TUI.

## Testing

- `tests/test_discovery_ollama_http.py`: payload /api/tags → modelli estratti; server
  giù → `available=False`; URL custom → usato nella GET (monkeypatch `_get_json`);
  lista vuota → "OK (no models listed)".
- `tests/test_config_threading.py`: roundtrip `ollama_base_url` in config.json; default
  quando assente; threading fino a `AnalysisConfig` via `build_analysis_config`.
- Smoke UI: `SettingsResult` con il nuovo campo.
- Adeguamento test esistenti: `tests/test_discovery_ds4.py` monkeypatcha
  `_discover_ollama` con lambda senza argomenti — le lambda vanno aggiornate alla nuova
  firma `(base_url)`.
- Verifica manuale sotto PTY (`script -qec`): boot con default (localhost), scan via
  Ollama locale, F2 con i due campi endpoint, provider line con `ollama(down)` a server
  spento.

## Versioning

Nuova funzionalità retrocompatibile → **minor bump a 0.11.0** (VERSION + pyproject a
mano) + CHANGELOG. Docs: README (sezione LLM Provider: endpoint configurabile),
PROJECT_SPEC (configurable items), CLAUDE.md (nota su discovery HTTP).

## Addendum (2026-08-02): troncamento Ollama

In fase di verifica e2e contro un server Ollama remoto con un modello 8B è emerso che i
tetti `num_predict` (facts 400, classify 320, repair 220, normalize 220) erano tarati su
modelli da 1B: l'output veniva troncato a metà JSON e, poiché `done_reason: "length"` non
veniva letto, il risultato degradato passava per valido. Il branch include quindi anche:
tetti ridimensionati (facts 2000, classify 1200, repair 1500), budget di normalize
proporzionale al batch (`max(800, 300 * batch_size)`), e `done_reason == "length"` trattato
come errore così scattano i fallback per-candidato. Dettagli e motivazione empirica nel
Task 5 di `docs/superpowers/plans/2026-08-01-ollama-configurable-url.md`.
