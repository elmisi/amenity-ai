# Design: backend LLM OpenAI-compatibile "ds4"

Data: 2026-08-01
Stato: approvato (design), in attesa di piano di implementazione

## Contesto

amenity-ai oggi usa un solo provider LLM: Ollama, chiamato da `analyzer.py` tramite le
funzioni module-level di `ollama_client.py` (`from .ollama_client import generate`).
Il protocol `LLMBackend` (`llm_backend.py`) esiste già ed è pensato per nuovi provider.

Va integrato un secondo provider locale: un server OpenAI-compatibile sulla rete
dell'utente (endpoint configurabile). Il server di riferimento espone (verificato via
probe):

- `GET /v1/models` → `deepseek-v4-flash`, `deepseek-v4-pro` (owner `ds4.c`, context 262k)
- `POST /v1/chat/completions` senza autenticazione
- Modelli *reasoning*: la risposta separa `reasoning_content` da `content`, ma solo se il
  budget di token è sufficiente; con `max_tokens` troppo basso il ragionamento finisce
  troncato dentro `content` (`finish_reason: "length"`)
- `response_format` JSON accettato ma **non applicato**
- `supported_parameters` include `reasoning_effort`, `max_tokens`, `temperature`, ecc.

## Decisioni (dalle domande di chiarimento)

1. **Coesistenza**: Ollama e ds4 attivi insieme; i modelli ds4 compaiono con prefisso `ds4:`.
2. **Vision**: ds4 è solo testo; le immagini restano sui modelli vision di Ollama.
3. **Priorità in "auto"**: `ds4:deepseek-v4-flash` è il primo candidato per facts e classify;
   i modelli Ollama restano come fallback.
4. Nessuna API key (il server non la richiede).

## Non-obiettivi

- Nessun supporto vision su ds4.
- Nessuna gestione API key / autenticazione (aggiungibile in futuro come campo config).
- Nessun refactor di `analyzer.py` verso dependency injection del backend (approccio B,
  scartato: troppa churn per lo stesso risultato).
- L'aggiornamento dipendenze è un task separato, fuori da questa spec.

## Architettura (approccio A: router per prefisso)

Due moduli nuovi; `analyzer.py` cambia solo import e pass-through di un campo config.

### `archiver/openai_client.py` — `Ds4Backend`

`Ds4Backend(BaseLLMBackend)` implementa `generate()` del protocol via
`POST {base_url}/v1/chat/completions` usando stdlib `urllib` (stesso stile di
`ollama_client._post_json`, nessuna dipendenza nuova).

Mapping dei parametri della pipeline:

| Parametro                     | Comportamento su ds4                                         |
|-------------------------------|--------------------------------------------------------------|
| `prompt`                      | `messages: [{"role": "user", "content": prompt}]`            |
| `think=False`                 | `reasoning_effort: "low"`                                    |
| `options["num_predict"]`      | ignorato; `max_tokens: 8000` fisso (il reasoning consuma completion tokens; i cap Ollama 220–400 troncherebbero a metà ragionamento. Il valore iniziale 1500 si è rivelato insufficiente in verifica e2e: il prompt di normalize ha prodotto `finish_reason: length`; inoltre `finish_reason == "length"` viene ora trattato come errore esplicito, così scattano i fallback esistenti) |
| `options["temperature"]`      | pass-through (`temperature: 0` nella pipeline)               |
| `response_format`             | non inviato (il server non lo applica; JSON garantito da prompt + estrazione/riparazione esistente nel normalizer) |
| `keep_alive`                  | ignorato                                                     |
| `images_b64`                  | `LLMResponse(error="ds4: vision not supported")` (difensivo) |

Lettura della risposta: **solo** `choices[0].message.content`; `reasoning_content` è
ignorato. `content` vuoto → `LLMResponse(error=...)`. Errori HTTP/timeout/JSON →
`LLMResponse(error=f"{type}: {msg}")`, mai eccezioni verso il chiamante (pattern identico
a `OllamaBackend`).

### `archiver/llm_router.py` — routing per prefisso

Costante `DS4_PREFIX = "ds4:"`. Espone `generate(...)` e `generate_with_image_file(...)`
con la stessa firma e lo stesso tipo di ritorno delle funzioni module-level di
`ollama_client.py`, più il parametro `ds4_base_url: str = ""`:

- `model.startswith("ds4:")` → `Ds4Backend(ds4_base_url).generate(model=model[4:], ...)`
- altrimenti → delega invariata a `ollama_client.generate(...)`
- `generate_with_image_file` con modello `ds4:*` → errore (mai vision su ds4)
- `ds4_base_url` vuoto con modello `ds4:*` → errore esplicito

Il prefisso `ds4:` è la convenzione di routing: viaggia con il model-id ovunque
(candidati, settings, `model_used` in cache, dettagli UI), quindi non serve altro stato
per sapere da quale provider viene un modello. Da documentare in CLAUDE.md e AGENTS.md.

### Modifiche a `analyzer.py`

- `from .ollama_client import generate` → `from .llm_router import generate`
- `AnalysisConfig` guadagna `ds4_base_url: str = ""`
- Tutti i call site di `generate(...)` (facts, classify, JSON repair) passano
  `ds4_base_url=config.ds4_base_url`. Il repair usa lo stesso modello che ha prodotto
  l'output: se è un `ds4:*`, il router instrada anche il repair su ds4. Nessun altro
  cambiamento di logica.
- Il percorso immagini (`extractors/image.py`) resta su Ollama, invariato.

## Discovery e selezione modelli

### `discovery.py`

- Nuovo `_discover_ds4(base_url)` → `GET {base_url}/v1/models`, timeout ~2.5 s, via
  urllib. Successo → `ProviderInfo(name="ds4", available=True, models=("ds4:deepseek-v4-flash", "ds4:deepseek-v4-pro"), details="OK")`.
  Il prefisso si applica qui, alla fonte.
- Errore/timeout → `ProviderInfo(name="ds4", available=False, details=...)`; nessuna nota
  rumorosa in UI.
- `discover_providers(*, ds4_base_url: str = "")`: stringa vuota = ds4 disattivato
  (il provider non viene nemmeno interrogato).

### `model_selection.py`

- `pick_model_candidates` raccoglie i modelli anche dal provider `ds4`.
- Candidati **testo**: `ds4:deepseek-v4-flash` e `ds4:deepseek-v4-pro` in testa alle
  preferenze (`_TEXT_PREFER`), prima della catena Ollama attuale. I filtri
  embed/vision non devono scartare i `ds4:*`.
- Candidati **vision**: i `ds4:*` sono sempre esclusi.

### Teste di lista coerenti

- `task_builders.build_analysis_config()` (facts, `prefer_fast`): `ds4:deepseek-v4-flash`
  primo, poi `ds4:deepseek-v4-pro`, poi la lista attuale.
- `app._ordered_classify_models()`: stesso ordine.

Risultato in "auto": facts e classify usano `ds4:deepseek-v4-flash`. Se il server ds4 è giù
al momento della discovery, ds4 non produce candidati; se cade a metà scansione, il
fallback per-candidato esistente passa al modello successivo (Ollama).

## Config, Settings e UI

- `config.AppConfig.ds4_base_url: str = ""` — persistito in
  `~/.config/amenity-stuff/config.json`, parsing difensivo in `load_config()` come gli
  altri campi stringa. **Il default è vuoto = feature disattivata**: il repo è pubblico
  e l'endpoint è una configurazione personale dell'utente, quindi nessun hostname va
  hardcodato nel codice o nella documentazione (negli esempi si usa
  `http://localhost:8000`). L'utente imposta il proprio endpoint una volta via F2 e
  questo resta solo nella config locale.
- `settings.Settings.ds4_base_url: str = ""` — wiring in `__main__.py` da `AppConfig`.
- `analyzer.AnalysisConfig.ds4_base_url` — threading via
  `task_builders.build_analysis_config()`.
- `settings_screen.py`: campo di testo "ds4 endpoint (OpenAI-compatible)"; vuoto =
  disabilitato. I modelli `ds4:*` sono selezionabili per facts/classify (nei
  campi/liste modello esistenti, qualunque sia il widget attuale), mai per vision.

## Error handling

- Server giù in discovery → provider non disponibile, candidati solo Ollama.
- Errore/timeout a runtime → `LLMResponse.error` → il chiamante esistente prova il
  candidato successivo; nessuna eccezione attraversa il TUI.
- `finish_reason: "length"` con `content` non-JSON → il normalizer/JSON-repair esistente
  fa il suo lavoro; se fallisce, si passa al candidato successivo.
- Timeout invariati rispetto a Ollama (180 s per facts/classify, 60 s per repair).

## Testing (manuale — non esiste una test suite)

Sotto PTY (`script -qec "..." /dev/null`), su una cartella campione:

1. Server ds4 attivo: scan + classify in "auto" → verificare `model_used =
   ds4:deepseek-v4-flash` nei dettagli e JSON facts valido.
2. Server spento: discovery senza ds4 → i candidati tornano Ollama; nessun errore in UI.
3. Server spento a metà scansione → fallback per-candidato, la riga non resta appesa.
4. `amenity-ai report` → timing plausibili per le voci ds4.
5. Settings screen: modifica endpoint, persistenza in config.json, tendine corrette.

## Versioning

Nuova funzionalità retrocompatibile → **minor bump a 0.10.0** (VERSION + pyproject.toml
aggiornati a mano; `scripts/bump_version.py` fa solo patch). CHANGELOG.md aggiornato.
