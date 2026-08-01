# Ristrutturazione della gestione dei provider LLM locali

Data: 2026-08-01 · Versione di partenza: 0.11.0 · Versione di arrivo: 0.12.0

## Obiettivo

Tre richieste, una sola ristrutturazione:

1. Aggiungere **vLLM** come terzo provider LLM
2. Ridurre la configurazione dei provider a **tre sole URL**, con scoperta automatica dei modelli
3. Aggiungere una modalità **doctor** che verifica se i modelli disponibili bastano a lavorare (almeno uno semantico e uno vision) e sa rimediare

## Stato di partenza

`archiver/` supporta due provider: Ollama (nativo) e "ds4" (un server OpenAI-compatible). Il routing avviene per prefisso del model-id (`ds4:` → `openai_client.Ds4Backend`, tutto il resto → Ollama). La scoperta è in `discovery.py`, la selezione in `model_selection.py`.

Problemi concreti che questa spec risolve:

- Le liste di preferenza dei modelli sono **triplicate** e quasi identiche: `model_selection._TEXT_PREFER`, `task_builders.prefer_fast`, `app._ordered_classify_models.prefer`. Nessuna delle tre conosce i modelli effettivamente in uso oggi.
- Le capability (vision, embedding) sono **indovinate dal nome**, con falsi negativi verificati sul campo (vedi *Verifica sul campo*).
- Non esiste modo di sapere perché una scansione fallisce quando manca un modello adatto.
- `openai_client.py` rifiuta categoricamente le immagini, quindi un modello multimodale servito via OpenAI API è inutilizzabile.

### Verifica sul campo (2026-08-01, `amarcord.local`)

Dati reali raccolti durante la progettazione, usati come fixture nei test:

| provider | endpoint | esito |
|---|---|---|
| Ollama 0.31.1 | `:11434` | `qwen3:8b`, 8.2B, Q4_K_M, ctx 40960, capabilities `["completion","tools","thinking"]` |
| vLLM 0.21.0 | `:8000` | `qwen3.6-27b`, root `/models/Qwen3.6-27B-AWQ-INT4`, `max_model_len` 131072 |
| ds4 | non configurato | 232B (`deepseek-v4-flash`), text-only, **mutuamente esclusivo** |

Due scoperte hanno cambiato il design:

- **`/api/tags` di Ollama ≥ 0.31 restituisce già `capabilities`, `parameter_size`, `context_length` e `digest`.** Non serve una `/api/show` per modello, quindi la scoperta resta a **una richiesta per provider** e non serve progettare alcuna cache delle capability.
- **`qwen3.6-27b` accetta immagini** (probe con PNG 1×1 → HTTP 200), pur non avendo nel nome né `VL` né `vision` né `llava`. Qualunque euristica sul nome lo avrebbe classificato text-only: è un falso negativo su hardware reale, ed è la giustificazione empirica del probe attivo.

## Decisioni

Ogni riga è una decisione presa esplicitamente durante l'intervista di progettazione.

| # | Decisione |
|---|---|
| 1 | Tre slot fissi in UI e config; internamente un registry di provider con `kind`, e un unico backend OpenAI-compatible condiviso da ds4 e vLLM |
| 2 | Le tre URL sono l'unico input richiesto; le tendine di override restano, popolate **solo** con modelli realmente scoperti |
| 3 | Quando manca un modello: catalogo curato + installazione in-app via `POST /api/pull` (solo Ollama, unico provider con un'API di installazione) |
| 4 | Capability: dichiarate dove esistono (Ollama), euristica sul nome altrove, **probe attivo dentro il doctor** con esito memorizzato |
| 5 | Doctor: logica pura in `doctor.py`, due superfici sulla stessa logica (`amenity-ai doctor` e tasto `[d]` nella TUI) |
| 6 | Scope del doctor: **solo provider e modelli**. OCR, dipendenze di estrazione e ambiente restano fuori |
| 7 | Ranking unico parametrizzato per ruolo, guidato dai metadati; le tre liste hardcoded spariscono |
| 8 | Taglie: `facts` → più piccolo, `classify` → vicino a ~8B, `vision` → più piccolo. Ordinamento a **fasce**, non a taglia esatta |
| 9 | Prefisso esplicito per tutti i provider (`ollama:`, `ds4:`, `vllm:`), con migrazione delle config esistenti |
| 10 | Priorità fra provider: **`vllm > ollama > ds4`**, e il provider è il **primo** criterio di ordinamento |
| 11 | Il doctor non si apre mai da solo; la riga di stato provider resta il canale passivo |
| 12 | La lista curata parte dai modelli già provati in passato ed è mantenuta a mano, con task dedicate |

### Nota sulla decisione 10

La priorità è stata rivista due volte durante la progettazione, sulla base di dati emersi dopo la scelta iniziale.

Il primo ordine scelto era `vllm > ds4 > ollama`, sul presupposto "server dedicato = più veloce", **e come terzo criterio di spareggio**. Applicato ai modelli reali, quel disegno produceva un risultato sbagliato: con `ollama:qwen3:8b` in fascia 5–9B contro `ds4:…` (232B) e `vllm:qwen3.6-27b` (27B) entrambi in fascia 20B+, la fascia decideva per prima e **Ollama vinceva sempre sul testo**. Sarebbe stata una regressione rispetto alla 0.11.0, dove ds4 è in cima a tutte e tre le liste di preferenza e `discovery.py:92` lo sceglie prima di Ollama.

Due fatti hanno poi determinato l'ordine finale:

- La taglia **non** approssima la velocità fra motori diversi: `deepseek-v4-flash` è un 232B "flash" (verosimilmente MoE con pochi parametri attivi), `qwen3.6-27b` è AWQ-INT4 su GPU con batching continuo. La taglia resta un buon criterio **dentro** un provider, che è l'unico posto dove convivono modelli non scelti uno per uno — cioè Ollama.
- **ds4 è mutuamente esclusivo**: se sta già rispondendo non può fare altro. Poiché `facts` e `classify` fanno una chiamata per file ciascuno, una scansione lunga lo monopolizzerebbe, e non esiste un ruolo "leggero" da riservargli. vLLM regge invece la contesa, e scalerà anche quando la scansione verrà parallelizzata.

## Architettura

### Moduli nuovi

| modulo | responsabilità | dipendenze |
|---|---|---|
| `providers.py` | registry `ProviderSpec`, priorità, `split_model_id` / `join_model_id` | nessuna (puro) |
| `capabilities.py` | euristiche sul nome, parsing della taglia, probe vision | nessuna I/O propria (il probe è iniettato) |
| `model_catalog.py` | rosa curata dei modelli installabili per ruolo | nessuna (puro) |
| `ollama_admin.py` | `pull_model()` in streaming, cancellabile | rete |
| `doctor.py` | `run_doctor()` → `DoctorReport` | puro, probe iniettato |
| `doctor_screen.py` | resa del report, selezione, progresso del download | Textual |

### Moduli modificati

| modulo | modifica |
|---|---|
| `discovery.py` | riscritto: tre sondaggi in parallelo, restituisce `ModelInfo` |
| `model_selection.py` | riscritto: `rank_models(models, role)` sostituisce `pick_model_candidates` |
| `llm_router.py` | risoluzione per prefisso invece di casi speciali; firma con `provider_urls` |
| `openai_client.py` | `Ds4Backend` → `OpenAICompatBackend`, con supporto immagini |
| `config.py` / `settings.py` | `providers: dict[str, str]` sostituisce i campi piatti; migrazione |
| `settings_screen.py` | tre campi URL; tendine da quattro a tre |
| `task_builders.py` / `app.py` | usano `rank_models`; sparisce `_ordered_classify_models` |
| `__main__.py` | sottocomando `doctor` |

### Registry dei provider

```python
@dataclass(frozen=True)
class ProviderSpec:
    name: str          # "vllm" | "ollama" | "ds4"
    kind: str          # "openai_compat" | "ollama"
    prefix: str        # "vllm:" | "ollama:" | "ds4:"
    default_url: str   # "" tranne ollama
    sends_reasoning_effort: bool = False
    supports_install: bool = False

PROVIDERS = (  # l'ordine di dichiarazione È la priorità
    ProviderSpec("vllm",   "openai_compat", "vllm:",   ""),
    ProviderSpec("ollama", "ollama",        "ollama:", "http://localhost:11434",
                 supports_install=True),
    ProviderSpec("ds4",    "openai_compat", "ds4:",    "",
                 sends_reasoning_effort=True),
)
```

`split_model_id` e `join_model_id` sono le uniche funzioni autorizzate a manipolare il prefisso. `split_model_id` verifica il prefisso **contro l'insieme dei prefissi noti**, mai spezzando sul primo `:` incontrato: `"ollama:qwen3:8b"` → `(ollama, "qwen3:8b")`, mentre un id legacy `"qwen3:8b"` non deve produrre un provider inesistente `qwen3`.

`sends_reasoning_effort` e `supports_install` sono le due sole differenze di comportamento fra provider dello stesso `kind`, e vivono come dati nel registry anziché come rami nel codice.

### Scoperta

```python
@dataclass(frozen=True)
class ModelInfo:
    id: str                       # già prefissato: "ollama:qwen3:8b"
    provider: str
    capabilities: frozenset[str]  # {"completion", "vision", "embedding", ...}
    parameter_size_b: float | None
    context_length: int | None
    capability_source: str        # "declared" | "heuristic" | "probed"
```

Tre sondaggi in parallelo con `ThreadPoolExecutor`, timeout 2,5 s ciascuno: il costo peggiore resta **2,5 s complessivi**, non 7,5 s.

- **Ollama** → `GET /api/tags`. Su ≥ 0.31 `capabilities`, `parameter_size` e `context_length` arrivano da lì (`capability_source="declared"`). Su versioni più vecchie, fallback a `POST /api/show` per modello; se manca anche lì, euristica.
- **openai_compat** → `GET /v1/models`. Si leggono `id`, `max_model_len` e `root` quando presente. `root` è la fonte migliore per taglia e quantizzazione (`/models/Qwen3.6-27B-AWQ-INT4` dice più di `qwen3.6-27b`). `capability_source="heuristic"`.

### Capability

Tre gradini di affidabilità decrescente:

1. **`declared`** — dichiarate dal provider, autoritative. `embedding` e l'assenza di `completion` escludono un modello dai candidati di testo, sostituendo la blocklist per sottostringa `("embed", "whisper", "tts")` di `model_selection.py:54`.
2. **`heuristic`** — pattern sul nome (`VL`, `vision`, `llava`, `moondream`, `minicpm`, `bakllava`, `pixtral`, `internvl`) più il parsing della taglia. Il parsing deve estrarre **27 da `qwen3.6-27b`**, non 3.6: il numero conta solo se seguito da `b` a fine token.
3. **`probed`** — solo dentro il doctor, solo sui modelli con `capability_source == "heuristic"`: una POST con un PNG 1×1 in formato `image_url`. Su un setup di solo Ollama non parte mai.

Esito del probe: HTTP 200 → `vision` confermata; 400 con messaggio relativo alle immagini → text-only, confermato; timeout, 500 o messaggio non riconducibile → **non conclusivo**, `capability_source` resta `heuristic`. Il probe può smentire l'euristica, non può fingere una certezza che non ha.

Solo gli esiti **conclusivi** vengono memorizzati, in `~/.config/amenity-stuff/probe_cache.json`, con chiave `(url del provider, id nudo del modello)` e valore l'insieme di capability confermate. La scoperta lo consulta all'avvio e promuove `capability_source` a `probed` quando trova una voce: è ciò che permette al percorso veloce di sapere che `qwen3.6-27b` è multimodale senza rifare il probe. La voce viene invalidata quando il modello sparisce dalla scoperta di quel provider.

### Ranking

```python
def rank_models(models: Sequence[ModelInfo], role: str) -> tuple[str, ...]
```

| ruolo | capability richiesta | criterio di taglia |
|---|---|---|
| `facts` | `completion` | fascia crescente |
| `classify` | `completion` | distanza dalla fascia 5–9B |
| `vision` | `vision` | fascia crescente |

Ordinamento a quattro livelli:

```
1. priorità del provider        (vllm > ollama > ds4)
2. fascia di taglia             (0–2B · 2–5B · 5–9B · 9–20B · 20B+)
3. posizione in CURATED_BIAS
4. id completo, alfabetico      (determinismo)
```

Le fasce esistono perché con la taglia esatta due modelli non sono quasi mai in pari e la lista curata non entrerebbe mai in gioco. Le fasce sono indicizzate `0…4`; per `facts` e `vision` la chiave di ordinamento è l'indice stesso, per `classify` è il valore assoluto della differenza fra l'indice del modello e quello della fascia 5–9B (indice 2), quindi `5–9B` batte `2–5B` e `9–20B` a pari distanza 1, risolte dal criterio successivo. I modelli di taglia ignota finiscono **in coda** anziché essere presunti piccoli: su `facts` un errore in quella direzione costa minuti di attesa.

`CURATED_BIAS` fa match sull'id **senza prefisso**, così una voce copre lo stesso modello su provider diversi; le voci intrinsecamente legate a un provider restano scritte per esteso. Parte dall'unione delle tre liste attuali — `gemma3:1b`, `qwen2.5:3b-instruct`, `phi4-mini`, `qwen3:4b`, `qwen3.5:4b`, `ministral-3:3b`, `gemma2:2b`, `qwen2.5:7b`, `mistral`, `gemma3`, `moondream`, `llava:7b`, `minicpm-v`, `bakllava`, `ds4:deepseek-v4-flash`, `ds4:deepseek-v4-pro` — più `qwen3:8b` e `qwen3.6-27b`, oggi assenti.

Quella lista è tarata su hardware più limitato di quello attuale, il che con le fasce non è più un problema: la lista curata ordina **solo dentro una fascia** e non può far vincere un 1B su un 8B. È un artefatto mantenuto a mano e aggiornato con task dedicate; nessun auto-benchmark, nessun apprendimento automatico delle preferenze.

Esito sui modelli reali di oggi, con ds4 non configurato:

```
facts     → vllm:qwen3.6-27b
classify  → vllm:qwen3.6-27b
vision    → vllm:qwen3.6-27b   (solo dopo il probe; prima l'euristica lo dà text-only)
```

Con vLLM spento: `facts` e `classify` → `ollama:qwen3:8b`, `vision` → nessuno, segnalato dal doctor.

### Routing

```python
spec, bare_id = split_model_id(model)
url = provider_urls.get(spec.name, "")
if not url:
    return error(f"{spec.name}: endpoint non configurato")   # mai un fallback muto
backend = OllamaBackend(url) if spec.kind == "ollama" else OpenAICompatBackend(url, spec)
```

Cambia la firma pubblica di `llm_router.generate` e `generate_with_image_file`: i parametri `base_url` e `ds4_base_url` sono sostituiti da un unico `provider_urls: Mapping[str, str]`. È la modifica che si propaga più lontano (`analyzer.py`, `task_builders.py`, `app.py`) ed è meccanica.

`Ds4Backend` diventa `OpenAICompatBackend`, condiviso, con tre differenze rispetto a oggi:

- **Immagini supportate**: `content: [{type:"text"}, {type:"image_url", image_url:{url:"data:<mime>;base64,…"}}]`, MIME dedotto dai magic bytes del file.
- **`reasoning_effort: "low"` solo dove dichiarato** dal registry (ds4). Mandarlo a vLLM rischia un 400 su un campo che non conosce; il comportamento di ds4 resta identico a oggi.
- **`max_tokens`** parte dalla costante attuale (8000) e viene abbassato se il provider dichiara un `max_model_len` inferiore.

Resta la regola già documentata in `openai_client.py`: si legge **solo** `message.content`, mai i campi di ragionamento. Il probe su `qwen3.6-27b` ha confermato che questi modelli riempiono un campo `reasoning` separato lasciando `content` a `null` finché la fase di ragionamento non è finita; la guardia su `finish_reason == "length"` introdotta nella 0.11.0 intercetta correttamente il caso di budget insufficiente.

Il livello LLM resta **stateless, senza stato mutabile condiviso**, un backend istanziato per chiamata: quando la scansione verrà parallelizzata (fuori scope qui) andranno toccati solo il ciclo worker in `app.py` e il locking sulla scrittura della cache.

### Doctor

```python
@dataclass(frozen=True)
class Remedy:
    kind: str        # "pull" | "hint"
    model: str
    provider: str
    size_bytes: int  # indicativo, dal catalogo
    note: str

@dataclass(frozen=True)
class Check:
    key: str         # "provider.vllm" | "role.vision"
    label: str
    status: str      # "ok" | "warn" | "fail" | "skip"
    detail: str
    remedies: tuple[Remedy, ...] = ()

@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[Check, ...]
    @property
    def worst(self) -> str: ...   # "fail" > "warn" > "ok" > "skip"
```

Cinque check in due gruppi:

- **Uno per provider** — non configurato → `skip`; configurato ma irraggiungibile → `fail` con il motivo vero (`ConnectionRefusedError`, timeout, DNS); raggiungibile con zero modelli → `warn`.
- **Uno per ruolo semantico, uno per ruolo vision** — `ok` se `rank_models()` restituisce almeno un candidato con capability `declared` o `probed`; `warn` se l'unico candidato poggia su euristica, che è esattamente il caso in cui il probe serve; `fail` se non ce n'è nessuno. Un modello fissato a mano in config ma non più presente nella scoperta produce anch'esso un `warn`, con nome e provider mancante: oggi quel caso fallisce in silenzio.

`Remedy` è un dato, non una stringa. `kind="pull"` viene generato solo se esiste un provider con `supports_install=True` raggiungibile; altrimenti il doctor degrada da solo a `kind="hint"` con il comando da lanciare a mano. Per vLLM e ds4 è sempre `hint`, e il suggerimento è la flag `--model` da usare al riavvio del server, non un `pull` che su quei provider non esiste.

`model_catalog.py` è la rosa curata degli installabili per ruolo: tag Ollama, dimensione indicativa su disco, una riga di motivazione. Stesso criterio di manutenzione di `CURATED_BIAS`.

`ollama_admin.pull_model()` fa `POST /api/pull` in streaming di righe JSON (`status`, `completed`, `total`), seguendo le due convenzioni già in vigore nel progetto: cancellazione cooperativa via callback `should_cancel`, ed esecuzione in un worker perché non deve mai bloccare l'event loop di Textual. Il download avviene **sulla macchina che ospita Ollama**: puntando ad amarcord i gigabyte finiscono lì. La schermata mostra host e dimensione prima di partire; la selezione più `invio` è la conferma. A download riuscito rilancia la scoperta e ricalcola il report, così il check passa da rosso a verde senza riavviare.

**Superfici**: `amenity-ai doctor` apre la schermata ed esce con codice diverso da zero se `worst == "fail"`, così resta usabile in uno script di post-installazione; il tasto `[d]` nella TUI apre la stessa schermata. **Nessuna apertura automatica**: la riga di stato provider prodotta da `ui_status.provider_summary` resta il canale passivo.

## Migrazione

Tutta dentro `load_config()`, con la stessa tecnica già usata per il vecchio `text_model`:

| chiave 0.11.0 | esito |
|---|---|
| `ollama_base_url` | → `providers["ollama"]` |
| `ds4_base_url` | → `providers["ds4"]` |
| — | `providers["vllm"] = ""` |
| `facts_model: "qwen3:8b"` | → `"ollama:qwen3:8b"` |
| `facts_model: "ds4:…"` | invariato |
| `*_model: "auto"` | invariato |
| `vision_model_fallback` | ignorata |

`vision_model_fallback` sparisce come setting: serviva ad accodare a mano un secondo modello vision perché la lista era corta e indovinata. Ora la catena di fallback **è** la lista ordinata, che `analyzer.py` già percorre finché una chiamata non riesce. Le tendine in settings passano da quattro a tre.

`save_config()` serializza `config.__dict__`, quindi le chiavi vecchie spariscono al primo salvataggio: la migrazione è **a senso unico** e tornare alla 0.11.0 significa riconfigurare gli endpoint. Scelta consapevole, nessuna finestra di compatibilità.

Le `cache.json` esistenti non vengono migrate: `model_used` contiene nomi nudi ed è puramente informativo, letto solo da `perf_report.py`. Conseguenza accettata: nel report di performance un modello usato prima e dopo l'aggiornamento comparirà come due righe distinte.

## Test

`tests/`, pytest dev-only (deliberatamente fuori da `pyproject.toml`), nessun test sulla TUI — coerente con il progetto.

- **`providers.py`** — round-trip `split`/`join`; `"ollama:qwen3:8b"` spezzato solo sul prefisso noto; un id legacy `"qwen3:8b"` non deve produrre un provider `qwen3`.
- **`capabilities.py`** — taglia estratta da `qwen3.6-27b` (27, non 3.6), da `/models/Qwen3.6-27B-AWQ-INT4`, assente quando il nome non dice nulla; euristica vision sui nomi noti.
- **`rank_models`** — i quattro livelli di ordinamento, i tre ruoli, l'esclusione autoritativa degli `embedding`, i modelli di taglia ignota in coda.
- **`discovery`** — fixture con i payload reali catturati il 2026-08-01 da `amarcord.local` (Ollama 0.31.1 `/api/tags`, vLLM 0.21.0 `/v1/models`), più payload malformati, estendendo il test di rifiuto già presente dalla 0.11.0.
- **`run_doctor`** — provider finti: tutti spenti, solo Ollama, vision solo euristico, modello fissato inesistente.
- **`llm_router`** — routing per prefisso; provider non configurato produce un errore esplicito.

## Rilascio

MINOR: `0.11.0` → **`0.12.0`** — nuove funzionalità più la rimozione di `vision_model_fallback`. `scripts/bump_version.py` incrementa la patch, quindi `VERSION` e `pyproject.toml` vanno allineati a mano per la minor, mantenendoli in sincrono.

Documentazione da aggiornare: `CLAUDE.md` (le sezioni *Model Selection* e *Two Configuration Dataclasses* descrivono l'assetto a due provider), `PROJECT_SPEC.md`, `README.md`, `CHANGELOG.md`.

**Vincolo invariato**: nessun hostname reale nel repository. I default restano `http://localhost:11434` per Ollama e stringa vuota per vLLM e ds4; gli endpoint vivono solo nella config locale dell'utente.

## Fuori scope

Dichiarato esplicitamente, per non far crescere la spec:

- Parallelismo della scansione (il design lo abilita mantenendo il layer LLM stateless, ma non lo implementa)
- Auto-benchmark o apprendimento automatico delle preferenze sui modelli
- Check su OCR, dipendenze di estrazione, ambiente (archivio, tassonomia, disco): il `DoctorReport` è una lista estendibile di `Check`, quindi aggiungerli dopo è additivo
- Provider aggiungibili o rinominabili dall'utente
- Installazione di modelli su vLLM e ds4, che non espongono un'API adatta
