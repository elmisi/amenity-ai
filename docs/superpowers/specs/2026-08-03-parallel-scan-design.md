# Parallel scanning — design

**Goal:** run several files through the facts phase at once, bounded per provider, without
making the TUI feel out of control.

## Why, and how much it can possibly buy

Measured against the real vLLM (`qwen3.6-27b`, AWQ-INT4) with the app's own router and payload,
unique content per request so prefix caching cannot flatter the numbers:

| concurrency | throughput | speedup | median latency |
|---|---|---|---|
| 1 | 0.070 req/s | 1.00× | 14.6s |
| 2 | 0.125 req/s | 1.79× | 16.2s |
| **4** | **0.204 req/s** | **2.9×** | 19.1s |
| 6 | 0.215 req/s | 3.1× | 22.6s |
| 8 | 0.203 req/s | 2.9× | 28.3s |

The knee is at 4. Past it, throughput stops improving and per-file latency keeps growing; at 8 it
regresses. **Three times is the ceiling**, and it is a property of the server, not of the client.

Where the time goes today, from a real cache of 810 scanned files:

```
facts phase   23,835s   6h37m
  ├─ LLM       17,697s   74%
  ├─ extract    5,366s   23%   (2,051s of it OCR, over 152 files)
  └─ rest         772s    3%
```

Three quarters of the wall clock is a thread parked in `urlopen`. The remaining quarter is mostly
Tesseract and `pdftotext` — subprocesses, which do not hold the GIL and have 12 cores available.
So the unit of work parallelises end to end, and the expected result on that corpus is roughly
**6h37m → under 2h**.

Those proportions come from runs on Ollama with a 3B model; absolute times on the 27B differ. The
74/23 split is what transfers, and it is what decides the design.

## What is already in place

The provider redesign left the ground clean, which is why this is a small change:

- `llm_router.generate` builds one backend per call and holds no state; `urlopen` opens a
  connection per request and is thread-safe.
- `extract_facts_item(item, config)` is pure: data in, data out, no access to `self`.
- `ScanItem` is frozen, and every UI update goes through `call_from_thread` keyed by **path**
  (`app.py:452`), not by row index — so results arriving out of order cannot collide.
- The only shared mutable state in the LLM layer is `_default_backend` in `ollama_client.py:122`,
  reachable only from that module's legacy module-level `generate`/`generate_with_image_file`,
  which have no callers. They are removed here rather than made thread-safe.

Two loops are what actually serialise the work:

- `app.py:495` — `for it in list(self._scan_items):` around the synchronous call.
- `normalizer.py:304` — `for batch in _chunk(items, chunk_size):`, one classify chunk after another.

## Architecture

### The limit belongs to the provider, and is enforced in the router

ds4 is mutually exclusive: four concurrent requests to it is exactly the harm the
`vllm > ollama > ds4` priority exists to avoid. Since analysis walks a candidate list, two files
of the same run can land on different providers, so a pool sized "for the main provider" is wrong
the moment a fallback fires. The limit therefore has to be applied where the provider is known —
at the call site in the router.

A new `archiver/concurrency.py`:

```python
@dataclass(frozen=True)
class ConcurrencyLimiter:
    """One semaphore per provider. Immutable once built; the semaphores are the only
    mutable things, and they are what a semaphore is for."""

    slots: Mapping[str, threading.Semaphore]

    @classmethod
    def from_limits(cls, limits: Mapping[str, int]) -> "ConcurrencyLimiter": ...

    def slot(self, provider: str) -> ContextManager[None]:
        """Acquire a slot for `provider`; unknown names are not limited."""

    def limit(self, provider: str) -> int: ...
```

`llm_router.generate` and `generate_with_image_file` gain an optional
`limiter: Optional[ConcurrencyLimiter] = None`. The slot wraps **only** `backend.generate(...)`,
never prompt construction, and is released on exception because it is a context manager. With
`limiter=None` behaviour is exactly today's, so every existing test stays valid unchanged.

No module-level registry, no `configure()` call: the limiter is passed down beside
`provider_urls`, which already travels that whole path. The router stays as stateless as its
docstring claims.

### Configuration

`ProviderSpec` gains `max_concurrency: int = 1`. Registry defaults:

| provider | default | why |
|---|---|---|
| vllm | 4 | measured knee |
| ollama | 1 | we cannot see `OLLAMA_NUM_PARALLEL`; sending more only queues server-side and makes the UI claim work that is not happening |
| ds4 | 1 | mutually exclusive by nature |

`AppConfig` and `Settings` both gain `provider_concurrency: dict[str, int]`, normalised in
`__post_init__` exactly like `providers` is: unknown keys dropped, missing keys filled from the
registry, values clamped to `1..16`. Configs written before this version simply lack the key and
get the defaults — no legacy migration needed.

Settings screen: a narrow numeric field beside each provider URL.

```
┌ Settings ───────────────────────────────────────────────┐
│ Providers                                               │
│   vllm    http://…:8000              parallel: [ 4 ]    │
│   ollama  http://…:11434             parallel: [ 1 ]    │
│   ds4     (empty = disabled)         parallel: [ 1 ]    │
└─────────────────────────────────────────────────────────┘
```

Empty or non-numeric input falls back to the registry default rather than erroring.

### The parallel facts worker

`_run_extract_pending` replaces its loop with a `ThreadPoolExecutor` sized
`max(limiter.limit(p) for p in configured providers)`, capped at 16. The semaphores, not the pool,
are the real regulator: a thread that cannot get a slot simply waits, while the others extract.
Since extraction happens before the LLM call inside `extract_facts_item`, this pipelines naturally
— threads do OCR while their peers wait on the GPU.

Each task marks its own row `scanning` **when it starts**, not at submit time, otherwise every row
in the table would light up at once.

Per item, the call is wrapped in `try/except`. Today it is not (`app.py:491-529`): an exception in
extraction escapes the loop, `finish()` never runs, and the UI stays stuck on "running" forever.
With a pool that stops being a rare annoyance and becomes a way to lose a whole run, so a failing
item becomes `status="error"` with the exception type as reason, and the run continues.

Cancellation keeps today's semantics: `x` stops new work, in-flight requests cannot be interrupted
because `urlopen` cannot be. `executor.shutdown(wait=True, cancel_futures=True)` drops what is
queued and waits for what is running — so the stop takes up to K requests instead of one. That is
a real UX change and the banner has to say so.

### Cache writes

`apply_result` currently does `upsert()` + `save()` per file, and `save()` re-serialises the whole
cache. At four times the rate, on the Textual event loop, that starts stealing frames. A
`SaveThrottle` in `cache.py` (injected clock, so it is testable) writes at most every 5s or every
25 dirty entries, whichever comes first, and always on finish and on cancel.

Not SQLite. That is the right long-term answer and is deliberately left for later.

### Progress feedback

Progress is scoped to the **run**, not to the table — the table can hold items classified or moved
in earlier sessions, and mixing the two produces counters that do not add up. The worker owns
`total`, `completed`, `in_flight`, `skipped`, `error` and the timestamps of recent completions.

Pure functions in `ui_status.py`, so all of it is unit-testable without a UI:

```python
@dataclass(frozen=True)
class RunProgress:
    total: int
    completed: int
    in_flight: int
    skipped: int
    error: int

def compute_rate(timestamps: Sequence[float], *, now: float, window_s: float = 60.0) -> Optional[float]
def format_eta(seconds: Optional[float]) -> str          # "~1h04m left" / "~45m left" / ""
def progress_line(progress: RunProgress, *, rate: Optional[float], eta_s: Optional[float]) -> str
```

The rate is measured over a sliding 60s window so it reacts to a slow file instead of averaging it
away; before the window has data it falls back to the run average, and while it has none the ETA
segment is simply absent rather than wrong.

The progress line replaces the per-status counter line **only while a run is in flight**; when idle
the existing line comes back unchanged. `files:` stays the table total, while `queued`, `in flight`
and `done` count the run's own targets — on a table where some files are already classified those
two numbers legitimately differ, and conflating them is what produces counters that do not add up.

```
┌ banner ─────────────────────────────────────────────────┐
│ RUNNING: scanning — 4 in flight                         │
└─────────────────────────────────────────────────────────┘
files: 810 • queued: 794 • in flight: 4 • done: 12 •
skipped: 0 • error: 0 • 0.21 file/s • ~1h04m left
```

and on `x`:

```
┌ banner ─────────────────────────────────────────────────┐
│ STOPPING — waiting for 3 requests in flight             │
└─────────────────────────────────────────────────────────┘
```

### The banner tells the truth about providers

`ui_runtime.provider_problem` (`ui_runtime.py:67`) inspects **only** Ollama. With Ollama
deliberately stopped and vLLM covering both roles, the TUI shows a permanent red
`ERROR: Ollama is not available`. Adding progress feedback to a banner that is already lying is
pointless, so it is fixed here.

It is replaced by `runtime_problem(discovery, settings)`, which calls the existing
`run_doctor(..., probe=None)` — pure, no network, the discovery result is already in hand — and
maps roles to severity:

| condition | severity | message |
|---|---|---|
| no discovery yet | info | `Detecting providers…` |
| `role.text` fails | error | `No semantic model available` |
| `role.vision` fails | warn | `No vision model — images will be skipped` |
| a configured provider is unreachable, roles covered | warn | `ollama unreachable` |
| a role check warns (pinned missing, capability guessed) | warn | the check's own detail |
| otherwise | ok | none |

`banner_for_state` gains a `warn` severity: unlike `error` it does not replace the RUNNING banner,
it is appended to it — the same treatment `problem` already gets.

## Phases

**Phase 1 — the facts phase.** Everything above. This is where 74% of the time is.

**Phase 2 — classify chunks.** `normalize_items` runs its chunks through the same limiter. It is
separate because the restructuring is genuinely delicate: today a failing chunk falls back to
single items and `continue`s, and a failing single-item chunk returns immediately and abandons the
rest. Under parallelism "return immediately" has no natural meaning. New semantics: each chunk is
independent, results merge in input order, and the reported error is the first error among chunks
that produced nothing. On 810 files this turns 68 sequential calls into 68/4.

Phase 2 can ship as its own PR; phase 1 is useful alone.

## Out of scope

- Parallel archive/move — filesystem renames are fast, and serialising them keeps collision
  handling simple.
- SQLite for the cache.
- Parallelising the directory walk (`scan_files`) — it is not a measurable cost.
- Per-model rather than per-provider limits.
- Any automatic tuning of the limit. The knee was measured by hand and is a hand-maintained
  number, like `CURATED_BIAS`.

## Testing

Pure and unit-testable, no network:

- `ConcurrencyLimiter`: that N+1 concurrent entrants block until one leaves, that the slot is
  released on exception, that an unknown provider is not limited.
- Router: that the limiter is acquired for the resolved provider and only around the call; that
  `limiter=None` is unchanged behaviour.
- Config: normalisation, clamping, unknown keys dropped, missing keys defaulted, a pre-0.13 config
  loading with defaults.
- `SaveThrottle`: fires on the count trigger, on the time trigger, and on force.
- `compute_rate` / `format_eta` / `progress_line`: including the empty-window case.
- `runtime_problem`: one test per row of the severity table, driven by synthetic
  `DiscoveryResult`s.

The TUI itself has no automated tests and does not get them here. Verification is by hand under a
PTY (`script -qec … /dev/null`) against the real vLLM, on an isolated `XDG_CONFIG_HOME` and a
scratch source folder: a fixed set of ~40 files timed before and after, checking the speedup is in
the region the benchmark predicts, that `x` stops cleanly, and that the counters and ETA stay
coherent throughout.

## Version

New feature, Python code: minor bump to **0.13.0**.
