# Session handoff — AI Travel Planner

> **Purpose:** the complete record of this project across every working session,
> so work can resume after the folder is closed with nothing lost.
>
> **Sessions:** 2026-08-08 (rebuild) · 2026-08-09 (persistence, Kafka, Supabase,
> Docker, caching, cleanup) · 2026-08-09 later (weather + hotel keys,
> Supabase-everywhere, real-browser verification) · **2026-08-11 (multi-leg
> journeys — see §14)**
>
> **Read this file first**, then `README.md` (which holds the 45-issue audit of
> the original code).

---

## 1. TL;DR — where the project stands right now

**Everything the user asked for is built, and every component has been verified
by actually running it.** Nothing is left in an unknown state.

| Component | Status |
|---|---|
| FastAPI (15 endpoints) | ✅ verified live |
| LangGraph, 12 nodes, real `interrupt()` | ✅ verified live |
| Redis caching + stale-while-revalidate | ✅ verified (66 s cold → 5 s warm) |
| Kafka work queue | ✅ verified (consume + commit lag 0 + DLQ) |
| Postgres persistence (Alembic, 4 tables) | ✅ verified |
| **Supabase** (hosted Postgres 17.6) | ✅ live, migrated, trip ran end-to-end |
| **Docker compose** (9 services) | ✅ full stack healthy, trip ran inside it |
| Web scrapers (etrain, AbhiBus, Google Flights) | ✅ verified live |
| Streamlit UI incl. human review + history | ✅ driven through a full submit→approve→itinerary flow |
| Tests | ✅ **119 passing**, pyflakes clean, `pip check` clean |

**Only genuinely outstanding item:** rotate the exposed credentials (§10).

> ⚠️ **Superseded on 2026-08-11.** A routing audit found six real bugs that this
> table did not cover — the planner could not express a multi-leg journey and
> silently substituted a distant airport instead of saying so. All six are now
> fixed and verified live; **§14 is the current record.** Tests: **159
> collected, 152 pass, 7 skip**.

**Session 4 changes:**
- `WEATHER_API_KEY` and `SERPAPI_KEY` are now **set and verified live**. Weather
  and hotels no longer degrade; the budget has a real accommodation line.
- **Supabase is now the backend everywhere.** Compose no longer overrides
  `DATABASE_URL` per-service — `migrate`, `api` and `worker` all inherit it from
  `.env`. The local `postgres` container moved behind a `testdb` compose profile
  and exists only for `make test-db`; it does not start with the stack.
- The whole flow was driven **in a real Chrome browser** — the last item that
  had never been verified that way.

`TAVILY_API_KEY` remains unset; cab fares stay labelled as estimates.

---

## 2. What this project is

`AI-Travel-Planner`, cloned from `https://github.com/Ultramech/AI-Travel-Planner`
into `/Users/gauranggarg/Desktop/Travel-Planner-Project-Resume/AI-Travel-Planner`.

It plans trips: parses a request, searches flights/trains/buses/cabs
concurrently, pauses for human review, then generates a weather-aware
day-by-day itinerary with a real cost estimate.

It is a **resume/portfolio project** (the parent folder is literally
`Travel-Planner-Project-Resume`). That matters: the user wants it to demonstrate
real system-design competence, not merely to work.

---

## 3. Everything the user asked for, in order

### Session 1 (2026-08-08) — the rebuild

Verbatim scope, so nothing is dropped:

1. Project was Streamlit-only → make it "good level", not only Streamlit
2. **Add API endpoints using FastAPI**, backend calls via FastAPI
3. **Correct use of checkpointers**
4. **Add Redis caching** where needed
5. **Use Kafka** where needed
6. **Verify the LangGraph structure is completely correct and working**
7. **Check the webscraper part**, find real issues
8. **Find and list all current issues**, fix them
9. **Apply system design concepts**
10. Make it functionally correct

Follow-ups in the same session:
- "Have you done everything, made the FastAPI endpoints also?"
- "What about the database and all?"
- "Tell me what you did of the webscraping part? Make sure it actually works, do
  anything to check. Also tell me what API keys to provide"
- "For flights can you use some MCP Server instead of using the API Key"

### Session 2 (2026-08-09) — persistence and proof

- "Did you recall what we were doing, search out for that file and read that"
- "I asked you to make it completely good and working… introduce some concepts
  of system design also, like kafka, database add. So make sure till where the
  work has been done"
- "Cover all the gaps, I will tell you the supabase or whatever you want, do
  everything, I am sharing the GROQ_API_KEY also" → supplied a real Groq key
- Supplied Supabase REST keys, then (after being told those cannot open a
  Postgres connection) the actual database URI + password

### Session 3 (2026-08-09, later) — polish and hardening

- "Fix all the issues in current system, make sure everything works correctly…
  Make sure you do something of webscraping part, and like load the previously
  cached results if found in redis, cache it, will do automatically update the
  webscraping… make sure complete flow is working or not. You may start the
  docker if you want… Do make sure that you check out everything and remove the
  unnecessary files that are of no need in my project"
- "do everything" (when interrupted mid-audit — meaning: proceed without asking)
- "So, in .env what all things to set, and have you tested the complete
  application working or not"

### Decisions the user made when asked

| Question | Answer |
|---|---|
| What should the database layer cover? | **"Persistence + history"** — Postgres + SQLAlchemy async + Alembic; trips, itineraries, provider-call audit log, cost snapshots; list/search past trips; move the checkpointer to Postgres. **No users/auth.** |
| How should the unverified parts be verified? | **"Check each and every component"** |
| Which database? | **Supabase** (chosen over local Postgres / their existing instance) |

---

## 4. What was built

Complete rewrite from a single-file Streamlit script into a service.

```
app/
  config.py            pydantic-settings; ALL secrets from env
  logging_conf.py      structured logging (replaced print()/st.write)
  api/
    main.py            FastAPI factory, lifespan, request-id middleware
    routes.py          15 endpoints
    schemas.py         request/response validation at the edge
  core/
    cache.py           Redis + in-process LRU fallback, single-flight locking,
                       negative caching, stale-while-revalidate, @cached
    bus.py             Kafka producer, event fan-out, SSE subscribe
    events.py          TripEvent/TripStatus/EventType models
    resilience.py      retry + timeout + circuit breaker + gather_providers
                       + automatic provider-call auditing
    context.py         contextvar carrying trip_id into provider calls
  db/                  ── added session 2
    base.py            declarative base + naming convention
    models.py          trips, itineraries, provider_calls, cost_snapshots
    session.py         async engine; URL normalisation (Supabase/pgbouncer)
    repository.py      all SQL lives here; best-effort writes, honest reads
  graph/
    state.py           TripState with explicit reducers per field
    nodes.py           async, UI-free nodes
    builder.py         topology + routers + shutdown_graph()
    checkpointer.py    process-wide checkpointer (postgres/sqlite/redis/memory)
  providers/
    llm.py             one ChatGroq factory; ask/ask_json/choose/extract_json
    geo.py             deterministic IATA + railway-station resolution
    weather.py         WeatherAPI, one call for the whole trip
    flights.py         Amadeus (optional)
    flights_google.py  Google Flights — KEYLESS, the default
    flights_router.py  source selection + failover
    trains.py          etrain.info scraper
    buses.py           AbhiBus scraper
    cabs.py            distance-based estimates (+ optional Tavily)
    hotels.py          SerpAPI Google Hotels
    itinerary.py       WikiVoyage-grounded plan generation
    extractor.py       validated TripBrief
  scraping/driver.py   Selenium lifecycle, guaranteed teardown, headless
  services/planner.py  drives the graph (start/resume/get/history)
  workers/consumer.py  Kafka worker
alembic/               migrations (added session 2)
scripts/check_db.py    DATABASE_URL diagnostics (`make check-db`)
ui/streamlit_app.py    thin HTTP client + past-trip history browser
tests/                 119 tests
legacy/                original v1 files + mapping table (KEEP — README cites it)
```

Plus `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `Makefile`,
`requirements.txt`, `.env.example`, `.gitignore`, `pytest.ini`.

### The graph (12 nodes)

```
extract_brief
   └─► plan_transport_strategy ─► search_transport ─► recommend_transport
           ▲ (which modes)         (ALL modes in        (compare real data)
           │                        parallel)                   │
           └──── revise_transport ──── human_review_transport ⏸ ─┘
                                                │ approve
                                                ▼
                                   enrich (weather ∥ hotels)
                                                ▼
                                        build_itinerary
                                                ▼
                    ┌────────────── human_review_plan ⏸
      revise_plan / revise_hotels / revise_transport │ approve
                    ▼                                ▼
              (loop, capped at 3)               finalise ─► END
```
`⏸` = a real `interrupt()`. Plus a `failed` node for unparseable requests.

### The 15 endpoints

```
POST   /api/v1/trips                  submit (202 async by default)
GET    /api/v1/trips                  list & search past trips  ← session 2
GET    /api/v1/trips/{id}             current state
POST   /api/v1/trips/{id}/resume      answer a human-review interrupt
GET    /api/v1/trips/{id}/stream      SSE progress
GET    /api/v1/trips/{id}/history     checkpoint audit trail
GET    /api/v1/trips/{id}/providers   provider-call audit log     ← session 2
GET    /api/v1/trips/{id}/itinerary   markdown export
GET    /api/v1/providers/stats        provider reliability        ← session 2
GET    /api/v1/graph                  mermaid topology
DELETE /api/v1/cache/{namespace}      invalidate
GET    /health  /health/live  /health/ready
GET    /
```

### The database (4 tables, Alembic-migrated)

| Table | Holds | Why not just Redis |
|---|---|---|
| `trips` | one row per request, from 202 to final state | system of record; survives eviction and restart |
| `itineraries` | the plan, **versioned per revision** | "what did you propose before I asked for cheaper" becomes a query |
| `provider_calls` | every outbound call: latency, ok, cache hit, result count | turns "the scrapers are flaky" into a number |
| `cost_snapshots` | budget per revision, `Numeric(12,2)` | money; the original did float maths on `"₹5,000"` |

The trip row is written **before** the job reaches Kafka, so a request is never
lost between the 202 and a worker picking it up. `provider_calls` is populated
automatically by the existing `@resilient` decorator plus a contextvar — no
provider signature grew a `trip_id` parameter.

---

## 5. Issues found in the original code

**45 issues catalogued in `README.md` → "Issues found in the original code".**
Do not re-derive them. The most important:

- `llm` was **never defined** in `mainpy.py` → `NameError` in 3 nodes including
  the default fallback route
- **Human review did nothing, ever** — `st.button` inside a graph node returns
  `False` on first render, so the edge always defaulted to approved
- `input()` called inside a graph node
- Messages grew quadratically — nodes returned `[*state["messages"], new]` into
  an `add_messages` reducer, which *appends*
- Every hotel price read "N/A" — `float("₹5,000")` raised, bare `except` hid it
- `persons` passed positionally into the `time` parameter of `find_cabs`
- **Checkpointer was decorative** — new `MemorySaver()` per call + fresh
  `uuid4()` thread_id ⇒ nothing ever resumed
- `train_data` never called `driver.quit()` → leaked a Chrome process per lookup
- Bus times sliced `times[::2]`/`times[1::2]` → shifted rows, silently wrong
- 4 live API keys hardcoded and pushed to a public repo
- No `requirements.txt`, no `.gitignore`, `.env` committed

---

## 6. Bugs found in MY OWN work, and how

Recorded because the lesson generalises: **running it found what reading it did
not.** This is the single most useful pattern on this project.

### Session 1 — the rewrite

| Bug | How it surfaced |
|---|---|
| Guessed AbhiBus URL doesn't exist — 302s to homepage | running the scraper live |
| `DOWN+ENTER` on autocomplete picked the first item — Delhi→Jaipur actually searched **Hyderabad→Bengaluru** | printing the resulting URL |
| `min()` over ₹ figures returned **40** (a "Save ₹40" discount) as the fare | reading real scraped output |
| Operator names read promo banners ("Flash Offer : ₹36 off per seat!") | reading real scraped output |
| JS `.click()` doesn't register with React state; **native `.click()` required** | side-by-side A/B test |
| etrain renders headers and a `Su Mo Tu…` strip as `<td>` rows → scraped as "trains" | reading real scraped output |
| Google Flights returns ragged arrays → `TypeError` | DEL→GOI worked, DEL→BOM and BLR→DEL died |
| `parse_price("Rs. 4,500")` → `0.45` (the "." in "Rs." read as a decimal) | own test |
| Pydantic constraint fired before the validator that would have fixed it | own test |
| `gather_providers` accepted `overall_timeout` and ignored it | lint |
| Kafka producer leaked on failed bootstrap | log inspection |

### Session 2 — persistence and Supabase

| Bug | Detail |
|---|---|
| **The test suite wrote to the real database** | no `conftest.py` existed, so tests inherited `.env`; once `DATABASE_URL` was set, `make test` wrote trips into it — Supabase or production just as readily |
| **Unparseable requests were stored as `completed`** | `_fail_fast` routes to END, which is also how success ends, so "no pending interrupt" read as "finished fine" |
| **Provider calls logged with a NULL `trip_id`** | the trip row didn't exist yet when the providers ran, so real scrapes were unattributable |
| `provider_stats(since_hours=…)` accepted the window and ignored it | same class of bug as the old `overall_timeout` |
| **`.env` was still tracked by git** | `.gitignore` listed it, but **gitignore does not untrack files git already knows** — the rule was inert and any `git add -A` would have committed live credentials |
| **Alembic could not run with this password** | `config.set_main_option()` goes through configparser, which reads `%` as interpolation syntax; `@` becomes `%40` in a URI → "invalid interpolation syntax" before any migration ran |
| **Checkpointer silently fell back to SQLite** | API and worker both run `setup()` at startup and raced to insert the same `checkpoint_migrations` row; the loser crashed and dropped to a local file, so "postgres" in config meant per-process state |
| **`/health` reported the configured backend, not the active one** | so that fallback was invisible — it claimed `postgres` while writing to SQLite |
| **A just-submitted trip briefly reported `completed`** | between graph steps a checkpoint has state but no scheduled `next`, read as "finished". Reproduced: at t=6s a trip that had done zero work returned `completed` |

### Session 3 — Docker, caching, cleanup

| Bug | Detail |
|---|---|
| **`pip install -r requirements.txt` was broken** | `langgraph-checkpoint==2.0.9` cannot satisfy `langgraph-checkpoint-postgres`; the venv had been upgraded in place so only a clean Docker build exposed it. Pinned to 2.1.2 |
| **Dockerfile never copied `alembic/`** | the `migrate` service runs `alembic upgrade head`; without those files it exits instantly and every dependent service waits forever |
| **`bitnami/kafka:3.7` no longer exists** | Bitnami removed the tag → official `apache/kafka:3.9.0` (env vars `KAFKA_*`, not `KAFKA_CFG_*`); `seleniarm/*` → `selenium/standalone-chromium` |
| **compose bound host 5432** | occupied by another local Postgres → moved to **5434** |
| **no `.dockerignore`** | `.env`, the 648 MB venv and every notebook were sent to the daemon and baked into layers |
| **Empty scrape results were cached under the FULL TTL** | scrapers return `{"count": 0, "buses": [], "note": …}` when they find nothing — truthy and non-empty — so `_is_empty` said "real data" and pinned "no trains available" for **6 hours** after one transient failure |
| **The `cache` singleton reused a Redis client across event loops** | a client binds to its creating loop; in any multi-loop process every op then failed *silently* and callers fell through to stale data |
| **`TEST_DATABASE_URL` pointed at the application database** | wrote 65 fixture trips into real history (purged; Makefile now defaults to the `_test` database) |
| `use_container_width` deprecated | removal date 2025-12-31 already passed → `width="stretch"` |

---

## 7. Key technical gotchas (do not relearn these)

1. **LangGraph 0.2.62 does NOT put `__interrupt__` in the `ainvoke` return.**
   Only `graph.aget_state(config).tasks[].interrupts` shows it.
   `planner.pending_interrupt()` handles both conventions — use it.
2. **A checkpointer binds to the event loop that opened it.** Reusing a compiled
   graph across loops gives "threads can only be started once". Call
   `builder.shutdown_graph()` between loops — tests do this in a fixture.
3. **The Redis client binds to its loop too.** `cache.connect()` now rebuilds
   when the loop changes; do not "simplify" that away.
4. **AbhiBus needs permanent numeric city IDs** (`Delhi/344/Jaipur/241`). Cannot
   build a URL from names. Discovered once per route, cached 30 days.
5. **Native `.click()`, never JS click**, on React autocompletes.
6. **Station dataset uses U+2010 hyphen**, not ASCII `-`.
7. `Station_Name_8477.xlsx` — capital N. Lowercase works on macOS, breaks on Linux.
8. **Station categories are NSG1..NSG6 / SG1-3 / HG1-3.** NSG1 = major terminals.
9. **protobuf conflict:** `fast-flights` needs protobuf ≥6; Streamlit 1.41 pinned
   <6. Resolved by Streamlit→1.61, FastAPI→0.141, starlette 1.3.1. Do not downgrade.
10. **MCP does not remove API keys.** Every flight MCP server wraps
    Amadeus/Duffel/Kiwi. MCP is a tool-calling protocol, not a data source. The
    keyless win came from Google Flights.
11. **Supabase:** region is **`ap-northeast-2`**, and the direct
    `db.<ref>.supabase.co` host is **IPv6-only** and unreachable from this
    machine — the **session pooler on 5432 is required** (6543 is transaction
    mode and breaks the checkpointer). Password contains `@` → `%40` in the URI.
12. **Alembic + `%` in a password:** never use `config.set_main_option()`;
    `env.py` builds the engine directly.

---

## 8. Verification status — what is PROVEN and what is not

### ✅ Proven by actually running it

| Component | Evidence |
|---|---|
| 119 tests | all pass, stable across repeated runs |
| pyflakes / `pip check` | clean |
| FastAPI | booted live; 202 submit, 422 invalid, 404 unknown, SSE streaming |
| **interrupt/resume across processes** | started in one interpreter, resumed in a **fresh** one |
| **Kafka** | real broker; 3 partitions, consume + **commit with lag 0**, malformed message → `trip.dlq` with raw payload preserved, worker survived |
| **Postgres / Supabase** | PostgreSQL 17.6; migrations applied, drift check clean, full trip persisted |
| **Docker compose** | all 9 services healthy; trip ran inside it (₹17,700) via the **remote Selenium container** (6 sessions), both workers on the shared checkpointer |
| Redis cache | hit/miss/skip; **66 s cold → 5 s warm** on an identical request |
| Stale-while-revalidate | stale entry served in 2 ms, background refresh landed, next call got the fresh value |
| **etrain scraper (LIVE)** | Delhi→Jaipur: 19702 SAINIK EXPRESS, 19610 YNRK UDZ EXP … 10 real trains |
| **AbhiBus scraper (LIVE)** | 10 real buses on near-term dates |
| **Google Flights (LIVE, keyless)** | DEL→BOM 58 found 1.5 s IndiGo ₹12,174; DEL→GOI 7; BLR→DEL 37 |
| **Streamlit UI** | driven through the real widgets: filled form → **Plan my trip** → live plan → "👤 Your input is needed" → approve → itinerary rendered; history browser lists real trips |
| Graceful degradation | with **zero** credentials the graph still reaches a coherent review pause |

### ⚠️ NOT verified

| Component | Why |
|---|---|
| **Amadeus** | not needed — flights are keyless |
| **Kafka 3-attempt requeue branch** | the DLQ path is proven; a valid-but-failing job is hard to construct because the graph handles most failures internally |
| **Tavily cab fares** | no credential; fares stay distance-based estimates |

### ✅ Added in session 4

| Component | Evidence |
|---|---|
| **WeatherAPI (LIVE)** | Jaipur +5 days: Partly Cloudy, 31.1 °C / 25.8 °C, 12 % rain. Forecast horizon on this key measured at **14 days** (2026-08-09 → 2026-08-22) |
| **SerpAPI hotels (LIVE)** | Goa: Majestic Beach Comforts ₹3,155/night → ₹9,464 total; ITC Grand Goa ₹45,065. Both per-night and total reported |
| **Supabase as the only backend** | full trip ran with the local `postgres` container **stopped**; `/health` reported `database: ok`, `checkpointer: postgres` |
| **Real Chrome browser, full flow** | Delhi→Goa 2026-08-23→26, 2 travellers: submit → transport review `interrupt()` → approve → plan review `interrupt()` → approve → completed. Cost ₹90,076 (₹45,038 pp) |
| **Persistence of that run** | `trip_284c38f666c64e93`: 1 itinerary revision, 1 cost snapshot (67,612 + 9,464 + 13,000 = 90,076), **6/6 provider calls ok**, 11 checkpoints, history browser shows 4 stored trips |

---

## 9. `.env` — what to set

**Mandatory: exactly one.**

| Variable | Status |
|---|---|
| `GROQ_API_KEY` | ✅ set and verified live |

**Recommended, already set:** `DATABASE_URL` (Supabase). Without it the app runs
but keeps no history — writes no-op and history endpoints answer 503.

**Set and verified in session 4:**

| Variable | Gains | Where |
|---|---|---|
| `WEATHER_API_KEY` | weather in the itinerary | weatherapi.com — **must be a WeatherAPI.com key**, `app/providers/weather.py:29` hits `api.weatherapi.com`. Forecast horizon 14 days |
| `SERPAPI_KEY` | real hotel prices + hotel line in the budget | serpapi.com |

**Still EMPTY, optional:**

| Variable | Gains | Where |
|---|---|---|
| `TAVILY_API_KEY` | live cab fares instead of estimates | tavily.com |

⚠️ **`DATABASE_URL` must be the session pooler.** It was found pointing at the
direct host `db.<ref>.supabase.co`, which has an **AAAA record only** — no A
record, so `getaddrinfo` fails outright on this machine. Corrected back to
`postgresql://postgres.<ref>:<pw>@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres`.
This is the second time that trap has been hit; see §7.11.

**Deliberately empty:** `AMADEUS_CLIENT_ID`/`SECRET` (flights keyless),
`SELENIUM_REMOTE_URL` (compose sets it inside Docker).

Everything else — Redis, Kafka, checkpointer, scraper, TTLs — has working
defaults. `.env` is **untracked**; `tests/test_repo_hygiene.py` fails if any
secret file becomes tracked again.

---

## 10. ⚠️ Credentials to rotate — the one real outstanding action

1. **Four keys in public git history** (`Ultramech/AI-Travel-Planner`):
   WeatherAPI, SerpAPI, Amadeus id+secret, Groq. Moved to env during the
   rebuild, but **history still contains them**. Rotation unconfirmed.
2. **Shared in chat on 2026-08-09** (so exposed wherever that transcript lives):
   the `GROQ_API_KEY`, the **Supabase database password**, and
   `SUPABASE_SECRET_KEY` — service-role equivalent, **bypasses Row Level
   Security**, the most sensitive of all.

Rotating the DB password means updating one line in `.env`.

---

## 11. Environment and how to run

- Python **3.11** at `.venv/` (system `python3` is 3.14 with no deps). Run
  ad-hoc scripts as `PYTHONPATH=. .venv/bin/python script.py`.
- **Docker compose is the easiest way to run everything** and is verified.
  API `:8000`, UI `:8501`. **The database is Supabase, in containers too** — as
  of session 4 compose does not override `DATABASE_URL`. The local scratch
  Postgres is profiled off; start it only for tests:
  `docker compose --profile testdb up -d postgres` (host `:5434`).
- The Homebrew **redis, kafka and postgresql@16 services were STOPPED** to free
  6379/9092 for Docker. Restart with `brew services start redis`,
  `brew services start kafka`, and
  `pg_ctl -D /opt/homebrew/var/postgresql@16 start` (that one is on **5433**).
- **`make test-db` must target `travel_planner_test`**, never the app database.
- `.env`'s `DATABASE_URL` points at Supabase and **every service now uses it**,
  containerised or not. Only `make test-db` uses the local scratch Postgres.
- **Nothing has ever been committed.** The repo's `.git` is inside
  `AI-Travel-Planner/`. The user has not asked for a commit — do not create one
  unprompted.

```bash
docker compose up -d      # everything, verified working
make check-db             # diagnose DATABASE_URL (connection, TLS, schema)
make migrate              # alembic upgrade head
make test                 # 119 tests, never touches a real database
make test-db              # includes the Postgres integration tests
make api / worker / ui    # run pieces locally instead
make graph                # print the mermaid topology
```

**Files deleted in session 3** (all recoverable via `git checkout HEAD -- <f>`):
`main.ipynb`, `main3 - Copy.ipynb`, `Station.xlsx`, `travel_graph.png`,
`output/*.exe`, `.env.bak.*`, `data/checkpoints`, `__pycache__`,
`.pytest_cache`. **`legacy/` was kept** — the README's audit cites it and the
before/after is the point of a resume project.

---

## 12. Suggested next steps

Nothing is blocking. In rough priority order:

1. **Rotate the credentials in §10.** The only real outstanding action. The
   WeatherAPI and SerpAPI keys added in session 4 were pasted into a chat
   transcript, so they join that list.
2. **The UI's default start date is `today + 14 days`** (`ui/streamlit_app.py:391`),
   which is exactly **one day past** the 14-day forecast horizon — so a
   default-settings trip always shows "Beyond forecast horizon" and never
   displays weather. Consider `days=10`.
3. **Commit the work** — nothing has ever been committed and there are ~35
   untracked paths. Ask the user first; they have never requested it.
4. Tighten `allow_origins=["*"]` before any real deployment.
5. Optional polish: exercise the Kafka requeue branch; wire
   `langchain-mcp-adapters` so the LLM calls MCP tools agentically (a good
   resume talking point, still needs provider keys).

---

## 13. How this user likes to work

- **They want proof by execution, not claims.** "Make sure it actually works, do
  anything to check." Live runs have caught bugs code review missed in every
  single session — this is the highest-value habit on this project.
- **Be explicit about verified vs unverified.** An honest "these three things
  are untested and here's why" list has been well received every time.
- **Correct false premises** rather than going along with them (e.g. MCP does
  not remove the need for an API key; Supabase REST keys cannot open a Postgres
  connection).
- They ask direct scoping questions mid-task ("have you done everything?") and
  expect a plain, complete answer.
- They say "do everything" when they want autonomy — proceed without stopping to
  ask, and report at the end.

---

## 14. Session 5 (2026-08-11) — multi-leg journeys

**The question that started it:** "no direct flight exists between Aligarh and
Mumbai, then he needs to do a multi-part journey — does the code handle that?"

**It did not**, and the way it failed was worse than not answering: it produced
a confident wrong answer. Asked for **Shimla → Mumbai** it replied
*"FLIGHT IXC → BOM dep 13:55, ₹32,604"*. IXC is Chandigarh, a 113 km drive from
Shimla. The traveller was sent to an airport in another state with no plan and
no budget for reaching it — and the price was double.

### What was wrong (all found by running the real providers)

| # | Bug | Evidence |
|---|---|---|
| 1 | **Silent origin substitution.** `geo.resolve_airport` LLM-picks a nearby airport; `origin_airport_city` and `exact` were stored on the brief and **read by nothing**. | Aligarh→AGR (85 km), Shimla→IXC (113 km), Gangtok→IXB (125 km), Munnar→COK (130 km) |
| 2 | **Flight prices double-counted.** `price_inr` is already the group total; the prompt asked for a group total, so the LLM multiplied again. | DEL→BOM: ₹6,074 / ₹12,148 / ₹18,222 for 1/2/3 adults — exact multiples. Shimla→Mumbai 2 pax reported ₹32,604 against a true ₹16,302 |
| 3 | **`arrival_date` never updated** from the chosen transport — pinned to `start_date` at `nodes.py:99`. | 24 h 30 m Aligarh→Mumbai train: itinerary told "arrive Aug 25", truth Aug 26. Hotel booked a night nobody sleeps in |
| 4 | **Journey time dropped before the comparison.** The scraper captures `duration: "24:30H"`; the TRAIN fact line omitted it. | LLM saw `dep 15:45 arr 16:15` — a 24.5-hour train reading as 30 minutes |
| 5 | **Thin routes crashed the parser and reported as healthy.** | AGR→BOM `'NoneType' object is not subscriptable`, KUU→BOM `list index out of range`, both inside `fast_flights/parser.py`; retried 3× though deterministic; `provider_health` said `"ok"` |
| 6 | **Kolkata's rail terminus was never tried.** Top matches were Sealdah and Kolkata station; Howrah is filed under its own district. | Kolkata→Delhi found **0** trains; now finds **10** from HWH |

### What was built

- **`app/providers/journey.py`** — models the journey as `access → main → egress`
  legs. Resolves the gateway city, prices the surface hop with the existing cab
  and train providers, and feeds it into the recommendation, the budget and the
  itinerary. Legs run concurrently under a bounded budget (`LEG_TIMEOUT` 45 s,
  `ALL_LEGS_TIMEOUT` 75 s) because they run *after* the main fan-out and land
  directly on the user's wait. A leg that cannot be priced is still shown —
  an unpriced leg beats an invisible one.
- **`geo.resolve_railhead`** — mirrors `resolve_airport` for cities with no
  railway at all (Gangtok → New Jalpaiguri). `search_transport` now queries the
  railhead, so "not in the dataset" is no longer the only possible answer.
- **`_CITY_EXTRA_STATION_CITIES`** — deliberately short (Kolkata→Howrah,
  Hyderabad→Secunderabad, Delhi→Nizamuddin, Kochi→Ernakulam). Padding it
  dilutes the NSG category ranking and pushes a city's own terminals down.
- **Explicit price semantics** — `price_total_inr` / `price_per_person_inr` /
  `party_size` on both flight providers, and a prompt that says which is which.
- **`parse_duration_minutes` / `derive_arrival`** — duration is authoritative;
  the "arrival clock < departure clock ⇒ +1 day" heuristic silently truncates
  anything over 24 hours, which is exactly the case that exposed this.
- **Honest absences** — `no_inventory` vs `degraded` on the flight router, a
  one-way retry when a route has no *round-trip* inventory, `NO_OPTIONS <MODE>`
  lines in the prompt, and a per-mode breakdown when nothing at all is bookable.

### Verified live, after

| Route | Result |
|---|---|
| **Shimla → Mumbai** | Two numbered legs, "Shimla has no airport" stated, **₹18,272** (₹16,302 flight + ₹1,970 road) — was ₹32,604 with the road leg invisible |
| **Aligarh → Mumbai** | Train 14314, journey **24h 30m** shown, **arrives 2026-08-26**, check-in moved to the 26th with the reason given, itinerary 5 days not 6, Day 1 starts in the evening |
| **Gangtok → Kochi** | *"No bookable transport"* with a reason per mode — no longer invents a Bagdogra flight from the leg fact alone |
| **Kolkata → Delhi** | **10 trains** from Howrah (was 0) |
| **Delhi → Goa / Port Blair** | Unchanged, correct, ₹27,626 / ₹61,524 for 2 |
| **AGR→BOM, KUU→BOM** | 1.2 s and *"No airline flies AGR→BOM"* — was 3 retries and "search unavailable" |
| **Full graph** | Shimla→Mumbai through both `interrupt()`s to `finalise`: total ₹42,363 (₹21,181.50 pp), **0 errors** |

**Tests: 159 collected — 152 pass, 7 skip** (was 119 collected: 112 pass,
7 skip) — `tests/test_multi_leg.py` adds 40 covering
duration parsing, overnight arrival, leg planning, price labelling, the gateway
note, railhead substitution and the hallucination guard. pyflakes and
`pip check` clean.

### Round 2 — "does it try everything?"

Follow-up question: *"Suppose I want to travel from Aligarh to Patiala — does it
try everything?"* It did not. The run searched **bus and train only**; the
strategy LLM had dropped flight *and* cab before any data existed, and the
summary then stated *"no flights are available"* — a claim about AGR→IXC that
nothing had checked. That is the original "commit to a mode before seeing data"
bug, surviving in a smaller form inside its own replacement.

**Fixed by making the filter deterministic.** `plan_transport_strategy` now
takes the road distance (one hard-cached lookup, shared with the cab provider)
and applies written rules — `MIN_FLIGHT_KM` 250, `MAX_CAB_KM` 900,
`MAX_BUS_KM` 1500, `MIN_INTERCITY_KM` 20. The LLM can no longer veto a mode; the
traveller's own words (`"I'd rather fly"`) override every rule; and anything
skipped is recorded in `strategy["skipped"]` with a reason that reaches
`provider_health`, the prompt (`NOT_SEARCHED <MODE>`) and the UI. **"We did not
look" and "there is nothing" are now different sentences.**

Aligarh→Patiala after: **540 km, all four modes searched, nothing skipped.**
AGR→IXC was actually checked (genuinely no service), and the cab option the old
filter had discarded surfaced as the runner-up at **₹7,520** — an alternative
the traveller previously never saw.

The same sweep then caught a bug the first round had introduced: **Delhi → Port
Blair quoted ₹18,310 to drive Kolkata → Port Blair**, 1,370 km across the Bay of
Bengal, because the railhead resolver offered Kolkata for an island. A leg past
`MAX_SENSIBLE_LEG_KM` (400 km) is now marked `implausible`, has its fare
**deleted rather than shown**, emits `UNUSABLE_GATEWAY` instead of
`CONNECTING_LEG`, and contributes nothing to the total. A fare there is worse
than no fare — it invents a road.

**Tests now 159 collected — 152 pass, 7 skip.**

### Round 3 — the layers I had changed but never run

The API schema and the Streamlit UI had been edited and only reasoned about.
Running them found three more bugs, all invisible from the graph layer:

- **The flight card said "per person" under a group total.** `_flight_card`
  hard-coded the label while displaying `price_inr`, so a two-traveller
  Shimla→Mumbai search rendered "₹16,302 per person" for a ₹16,302 *total* —
  the same double-count as the prompt, one layer further out. Now reads
  "₹16,302 · total for 2 · ₹8,151 pp".
- **Machine-readable sentinels were shown to the traveller.**
  `ARRIVAL_DATE: … ARRIVAL_TIME: … TRANSPORT_COST_INR: 18272` sat under the
  recommendation and `ESTIMATED_ONGROUND_BUDGET_INR: 6500` under the itinerary.
  `_strip_sentinels` removes both, including the run-together one-line form.
- **The strategy caption repeated the skip list** already rendered beneath it.

**Verified in a real Chrome browser** against the API on `:8077` (Supabase
checkpointer, Redis and Kafka down so it degraded to in-process cache and inline
execution) and Streamlit on `:8511`: form submit → transport review, showing
"Connecting legs — part of your journey · Shimla → Chandigarh ~113 km · road
~₹1,970", "Not searched: cab … bus …" as a caption rather than an outage
warning, and the corrected fare labels. HTTP path separately driven end to end:
`POST /api/v1/trips` (200, 33 s) → two `resume` calls → `completed`, total
₹32,855, `connecting_legs` and `arrival_date` present in the JSON, zero errors.

### Still open

- The station dataset has **almost no Kerala coverage** (only "Kozhikkode"), so
  Kochi/Munnar rail is unavailable. Reported honestly as a dataset gap, not as
  "no trains exist".
- **AbhiBus discovery can exceed the 120 s fan-out budget** on some routes
  (seen on Delhi→Agra). Pre-existing; the bus result is dropped and reported as
  a timeout rather than silently missing.
- No **connection-time validation** — the plan states a connecting leg is
  needed but does not verify the road hop actually lands before the flight
  departs. That is the natural next step.
- Multi-leg is modelled for **access/egress to a gateway**, not for arbitrary
  hub-to-hub routing (e.g. IXB→CCU→COK when no direct flight exists).

---

## 15. Session 6 (2026-08-11) — "fix everything"

Brief: *"Fix out everything… there must not be anything remaining to fix… think
like a real user, add some more functionality."* Every item §14 left open is
closed, plus a set of bugs found only by running the parts that had never been
run.

### New functionality

| Feature | Why a real traveller needs it |
|---|---|
| **Train fare estimates** (`trains.estimate_fares`) | etrain lists no fares, so any train recommendation left `budget.transport_inr` **empty** and the trip total silently dropped its largest line. Distance-based IR slabs for SL/3A/2A/1A with a long-distance taper; measured within ~5% of real IRCTC fares on Delhi–Kolkata and Chennai–Delhi. Aligarh→Mumbai went from `None` to ₹1,340. |
| **Connection-time validation** (`journey.check_connection`) | The plan named a road leg but never checked it landed before departure. Now every option carries *"leave Shimla by 07:55"*, night departures are flagged, and unmakeable ones are refused with the date you'd have to travel instead. Uses the LLM's **driving time**, which was being requested and thrown away — 113 km of hill road is 4 hours, not 2. |
| **Return journey** | Nothing ever planned the way home. Return flights are searched on `end_date` and carried as their own budget line. |
| **Hub routing** (`flights_router._search_via_hub`) | When no direct service exists, connect through DEL/BOM/BLR/HYD/MAA/CCU. Gangtok→Kochi went from *"no bookable transport"* to **IXB→DEL→COK, 9h20m, ₹41,162**, explicitly marked `self_transfer` — two bookings, your risk. |
| **Graceful degradation without the LLM** | Groq's free tier hit its 100k daily token cap mid-run. `recommend_transport` returned only errors; `build_itinerary` returned a **blank plan while the trip reported "completed"**. Both now compose from the structured data already in hand — real fares, real timings, real dates — and record `degraded` in `provider_health`. |
| **Deterministic gateway table** (`geo._GATEWAYS`) | 40+ well-known Indian destinations → the airport and railhead people actually use. Removes the LLM from the common path entirely: a quota outage had been turning Gangtok into a city with apparently no air service, deleting its flights, legs and budget. |

### Bugs found by running it

- **`fast_flights` reads alliance metadata before the flights.** A route flown
  only by non-alliance carriers raised `IndexError` and we reported
  **"No airline flies DEL→COK"** — a route with dozens of daily departures.
  Now parsed directly from `payload[3][0]` with metadata optional; "no
  inventory" is claimed only after reading an empty flight slot.
- **Wrong IATA codes in the bundled dataset.** Coimbatore was `CIB` (Council
  Bluffs, **Iowa**) and Kozhikode `CCI` (Concordia, **Kansas**); Pune and
  Bhuntar were missing entirely.
- **The return leg was double-counted.** The outbound search asked for a
  round-trip fare *and* a separate return was added: Shimla→Mumbai read
  ₹34,656 against a true ₹16,302 + ₹1,970. Outbound is now priced one-way
  whenever a return is searched separately.
- **AbhiBus parsed every card on the page.** Each `element.text` is a browser
  round trip and a busy corridor lists 100+ services, so Delhi→Agra blew its
  85-second budget and lost the result. Capped at 40; **85 s timeout → 29 s
  with 10 real buses**. Discovery is separately bounded at 45 s.
- **`ProviderError: unknown failure`** — `str(asyncio.TimeoutError())` is the
  empty string, so the one failure whose cause we always know reported as the
  one we don't. Now `"timed out after 85s"`.
- **The Dockerfile did not copy `data/`** and `.dockerignore` excluded it, so
  the new station and airport supplements vanished in containers — Kerala rail
  and Jaipur's airport would have worked locally and silently failed in Docker.
  Exactly the `alembic/` bug from session 3, repeated.
- **The test suite made live network calls.** With `KAFKA_ENABLED=false` the API
  degrades to an in-process background task, so `POST /api/v1/trips` launched
  real Chrome, scraped live sites and called a paid LLM — one test took **122
  seconds**. Suite now **8 s** and touches nothing external.
- **The suite also read the developer's Redis**, so a geo test passed alone and
  failed in the suite once the Docker stack was up. `conftest` now points the
  cache at a dead address.
- Selenium had **no concurrency bound** while the fan-out grew to 4-6 browsers;
  `scraper_max_sessions` (default 4) now makes pressure show as waiting.

### Verified

**176 tests pass, 7 skip** (from 119 at the start of the day), stable across
repeated runs. pyflakes and `pip check` clean.

**Full Docker stack** — 7 services healthy, Kafka `ok`, Supabase checkpointer,
remote Selenium confirmed creating and cleanly quitting sessions. Trips
submitted over HTTP, consumed by the **Kafka worker** (not the inline
fallback), driven through both `interrupt()`s to `completed`:

| Route | Result |
|---|---|
| Shimla → Mumbai | leg Shimla→Chandigarh 4h ₹1,970 · out ₹18,272 · back ₹19,590 · total ₹60,537 · **0 errors** |
| Gangtok → Kochi | hub IXB→DEL→COK ₹41,162 · leg 4h ₹2,120 · back ₹27,678 · total ₹77,660 |
| Shimla → Goa | out ₹17,654 · back ₹19,936 · itinerary degraded to a dated skeleton when the quota ran out, and said so |

### Still open — honestly

- **Rotate the exposed credentials (§10).** Unchanged and unconfirmed; this
  needs the account owner, not the codebase.
- **Groq free tier is 100,000 tokens/day** and a handful of full trips exhausts
  it. Degradation is now graceful, but a real deployment needs a paid tier.
- Hub routing is **same-day, one stop, two hubs**. It does not search multi-day
  or two-stop itineraries.
- Train fares are **estimates**, not IRCTC quotes; there is no seat-availability
  or booking path.
- `full_iata_codes.csv` holds ~100 airports and had three errors in the ones
  checked. The supplement fixes those found; the rest is unaudited.

---

## 16. Session 6b — features built to be *defensible*, not just present

Brief: *"add functionality to impress the interviewer… think out of the box."*
Rather than more travel features, these were chosen around one theme that the
day's debugging had already established: **this system lied in ways nobody could
see, so make it prove itself.** Each has a story attached to a real bug.

### 1. Deterministic scorer + LLM-disagreement audit — `app/core/decision.py`

The model, over one day of running, multiplied a group fare twice, read a
24 h 30 m train as a 30-minute hop, recommended an airport in another state
without mentioning the drive, and asserted "no flights available" for a route
nothing had searched. **Every one is an arithmetic or bookkeeping error, not a
language error.** So the split is now explicit:

> the numbers are computed in `decision.py`; the model explains the choice.

Each option carries a real group cost and real door-to-door time (connecting
legs and check-in buffers folded in), plus penalties for self-transfer, night
starts, overnight arrivals and estimated prices. Cost and time are **normalised
across the candidate set** rather than converted through a made-up
rupees-per-minute rate. Weights come from the traveller's budget level.

Then `compare_with_model` checks whether the prose actually picked the
top-scoring option, and records the gap when it did not. Not a correctness
check — the model may be weighing something real — an **audit**.

*Live demo worth showing:* Delhi→Goa at `budget_level=low` ranks the **₹5,700
bus above the ₹14,846 flight**; at `high` the flight wins. Same data, different
traveller, and the reasoning is visible.

### 2. Decision provenance — `GET /api/v1/trips/{id}/rationale`

Every recommendation now ships its own working: each option scored, the weights
applied, which modes were skipped **and why**, which prices are estimates rather
than quotes, and whether the model agreed with the maths. Rendered in the UI as
a collapsed "Why this recommendation?" drawer that flags disagreement in the
title.

### 3. Chaos injection — `app/core/chaos.py`

The project claims Redis, Kafka and every scraper can fail without taking the
trip down. That was never tested — it was tested *by accident* when Groq's daily
cap ran out. Now:

    CHAOS_PROVIDERS=etrain,abhibus CHAOS_MODE=error|timeout|empty

Injected at the single `@resilient` chokepoint, so no provider knows it exists.
It refuses to arm when `APP_ENV=production`. `empty` mode simulates the
**dangerous** failure — a provider that succeeds and returns nothing.

*Verified live:* both scrapers killed, flights still returned 4 options, both
failures reported honestly in `provider_health`.

### 4. Flexible dates — `app/providers/flex_dates.py`

Fares move a long way over 48 hours and the planner had no way to say so.
Bounded, cached fan-out across ±3 days; **~2.7 s for 7 dates**. Savings under
₹500 are not advertised — nobody rearranges a trip for that.

*Live:* Delhi→Goa found ₹13,516 on the 24th against ₹14,846 on the 25th.

### 5. Provider contract tests — `tests/test_provider_contracts.py`

The `fast_flights` break produced a **silent wrong answer**, and no mock would
have caught it because a mock keeps its original shape forever. So real payloads
are recorded by `scripts/record_fixtures.py` and replayed through the real
parser. Three cases deliberately: flights **with** metadata, flights **without**
it (the case that broke), and a route with genuinely no service — because
"nothing flies here" and "we could not read the answer" must never collapse into
the same output.

### Verified

**205 tests pass, 8 skip** (from 176). pyflakes, `pip check` clean. 15
endpoints. Full Docker stack rebuilt and driven through the browser: return
journey, date-flexibility table and the rationale drawer all render, with
*"✅ The written recommendation matches the top-scoring option"*.

### Honest limits

- The scorer's weights and penalty constants are **judgement, not measurement**.
  They are defensible and documented, not tuned against outcome data.
- `compare_with_model` matches the model's pick by substring. Good enough to
  flag disagreement; it is a smoke alarm, not a parser.
- Fixtures capture Google Flights only. The Selenium scrapers have no recorded
  contract — their HTML is too large to store usefully.
