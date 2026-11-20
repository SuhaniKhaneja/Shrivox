# Shrivox

![CI](https://github.com/SuhaniKhaneja/Shrivox/actions/workflows/ci.yml/badge.svg)

A multi-tenant Hinglish customer support agent built on a stateful
LangGraph pipeline with RAG-grounded responses and safety guardrails.
Any business can register, upload its own knowledge base, and get an
AI support agent scoped entirely to its own data. This repo covers
Phase 1 (text-over-WebSocket, complete); the same graph and transport
contract is designed to extend to streaming voice (Phase 2) and
telephony via Twilio (Phase 3) without re-architecting.

## Why this exists

Most "GenAI chatbot" portfolio projects are a single prompt calling an
LLM. Shrivox is built around the actual hard parts of a production
support agent: routing decisions that must happen *before* an
expensive LLM call (escalation detection), grounding responses in a
real per-tenant knowledge base instead of letting the model improvise
policy details, catching hallucinated guarantees before they reach a
customer, and enforcing tenant isolation at the database layer, not
just in application code.

## Architecture

```
User message (JWT identifies business_id)
   |
   v
[input_guard]  deterministic rules (English, Hinglish, Devanagari), no I/O
   |  escalate / self-harm ----------------------------------> [escalate_response] --> END
   |  (fan-out, runs in parallel)
   +--> [classify_intent]   (Gemini, tiny call)
   +--> [retrieve_context]  (pgvector, business_id in SQL, top-3; short follow-ups
   |                         also search with the previous customer message)
   v  (join)
[triage]  classifier says "wants a human" --> [escalate_response] --> END
   |
   v
[generate_response] --> [safety_check] --safe--> END
                            |  unsafe (ungrounded promise, a figure not in the KB/conversation, an unsupported
                            |  refusal, or a figure that applies to a different place than the customer's)
                            v
                  [regenerate_response] (targeted instruction, ONE retry) --> [safety_check]
                            |  still unsafe
                            v
                     [safe_fallback] (hand-off to a human, never the unsafe text) --> END

Response -> Redis (session memory, atomic append) + Postgres (durable record) -> client
```

Built as a **LangGraph StateGraph**, not a linear function, because the
branches are genuinely conditional: an escalation-triggering message
must skip retrieval and generation entirely (latency + cost), and an
output that trips the safety check goes through a bounded retry loop
and is never sent if it is still unsafe. See `app/agents/graph.py` for the reasoning behind each
node.

### Why each component exists

| Component       | Choice                                                                        | Why                                                                                                                                                                                                      |
| --------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Orchestration   | LangGraph                                                                     | Conditional branches (escalation, safety retry) need explicit edges, not nested if/else, especially as later phases add barge-in/clarification loops                                                     |
| LLM             | Gemini 2.5 Flash                                                              | Unifies the whole stack on one provider (embeddings already use Gemini) with one free-tier budget to track; Flash is priced/benchmarked for fast, high-volume chat                                       |
| Embeddings      | Google `gemini-embedding-001`                                                 | Free via an AI Studio key (no billing setup), natively multilingual (handles Hinglish well), truncated to 1536 dims via MRL to match the schema                                                          |
| Vector storage  | pgvector, inside the same Postgres                                            | One fewer moving part than a dedicated vector DB, one transactional boundary shared with tenant data                                                                                                     |
| Multi-tenancy   | `business_id` on every tenant-owned table, filtered in the SQL `WHERE` clause | A missing tenant filter is a real data-leak bug class (one business's customer getting an answer sourced from another's private docs) — enforcing it in SQL means no code path can accidentally skip it |
| Auth            | JWT (business_id + user_id + role in the token)                              | The WebSocket and REST endpoints are separate connections that both need "which business is this" without a DB round-trip per request                                                                    |
| Session memory  | Redis                                                                         | Hot, short-lived, TTL'd — must survive reconnects without living in process memory                                                                                                                        |
| Durable history | Postgres (`calls` + `messages`)                                               | System of record for transcripts/analytics; split tables because dashboard queries and per-turn analytics queries have different access patterns                                                         |
| Transport       | WebSocket                                                                     | Voice phases stream audio frames over the same connection contract — building this phase on request/response would mean re-architecting the transport later                                             |

## Project structure

```
app/
  main.py              FastAPI app entrypoint, wires routers together
  config.py            Centralized settings (env-var backed)
  core/
    security.py          JWT issuance/verification, password hashing
    upstream.py            Gemini retry policy (timeouts, backoff, honours the server's retry delay)
    failures.py              Fixed client-facing error messages (no exception text leaks)
    ratelimit.py              Redis rate limits (login, register, ticket, chat)
    observability.py           Prometheus metrics
    log_redaction.py            Redacts the WebSocket ticket from server logs
  api/
    deps.py               Shared auth dependency (JWT -> CurrentUser)
    auth.py                Business registration + login
    documents.py            Knowledge-base upload/list/delete
    conversations.py         REST call history + WebSocket chat endpoint
  agents/
    graph.py               LangGraph state machine (the core agent)
    guardrails.py            Input/output safety checks
    scope.py                   Scoped-figure rules ("X for a listed group, Y for all other")
    llm_client.py             Gemini wrapper (retry + latency capture)
  retrieval/
    retriever.py             pgvector search + Google embeddings (RAG)
    chunking.py                Word-based chunking with overlap
  memory/
    session_memory.py          Redis-backed hot conversation state
  db/
    models.py                    Multi-tenant Postgres schema
    session.py                     Async session factory
    vector_index.py                  HNSW index + iterative-scan setup (pgvector >= 0.8)
  prompts/
    templates.py                    System prompt + intent/response templates
  evaluation/
    eval_dataset.json                 Labeled test cases (Hinglish + English)
    run_eval.py                        Evaluation harness (see below)
    rag_eval.py                         RAG evaluation: retrieval metrics, LLM-judged answers, threshold sweep
    rag_corpus.json                      Realistic evaluation corpus (52 entries)
    rag_dataset.json                      Small demo question set
    rag_dataset_hard.json                  61-question hard set
    language.py                             Reply-language detector used by the eval
    guardrail_eval.py                        Offline guardrail regression eval
    guardrail_cases.json                      Labelled guardrail cases (self-written)
data/knowledge_base/        Sample Hinglish support KB (JSON)
scripts/
  init_db.py                    Creates the pgvector extension + tables
  seed_demo_business.py           Seeds a demo tenant + admin user + KB
  test_chat.py                      Command-line chat smoke test (login -> ticket -> WebSocket)
  bench_vector_search.py              Vector search benchmark on a scratch database
docs/
  AUDIT.md                        Audit and improvement log, with what could not be verified
  bench_vector_search_40k.json      Saved benchmark result
tests/                       pytest suite: unit tests mock external calls; tests/integration use real Postgres + Redis
docker/Dockerfile
docker-compose.yml           Postgres (pgvector) + Redis + app
.github/workflows/ci.yml     Test + build on every push
```

## Setup

```
cp .env.example .env
# fill in GOOGLE_API_KEY (aistudio.google.com/apikey) -- used for both reasoning and embeddings
# generate a JWT_SECRET, e.g.: openssl rand -hex 32

docker compose up -d postgres redis
pip install -r requirements.txt

python scripts/init_db.py
python scripts/seed_demo_business.py   # prints a demo login + business id
```

Run the server:

```
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for interactive API docs. For a quick
end-to-end check from the command line, run `python scripts/test_chat.py`
(see "Authentication model" below for what it does).

## Tests

```
python -m pytest tests                                   # unit suite (mocked services); integration tests are skipped
python -m pytest tests/integration --run-integration     # integration suite, needs real PostgreSQL+pgvector and Redis
```

The integration suite (`tests/integration`) runs the real app against real
PostgreSQL (pgvector) and Redis. Only Gemini is faked (deterministic
bag-of-words embeddings, recorded chat prompts). It proves tenant isolation at
every SQL query site that touches tenant data, in pgvector retrieval, and in
Redis keys, using two tenants with *identical* knowledge so a missing
`business_id` filter cannot hide behind a weaker match.

To run it locally:

```powershell
docker compose up -d postgres redis
$env:GOOGLE_API_KEY = "dummy"; $env:JWT_SECRET = "ci_test_secret"
$env:POSTGRES_DSN = "postgresql+asyncpg://shrivox:shrivox@localhost:5432/shrivox_test"
$env:REDIS_URL    = "redis://localhost:6379/15"
python -m pytest tests/integration --run-integration -q
```

The suite truncates tables and deletes Redis session keys, so it refuses to
run unless the database name ends in `_test` and Redis is logical DB 15. A
missing `shrivox_test` database is created automatically. If you pass
`--run-integration` and the infrastructure is unreachable, the run fails
loudly instead of skipping. The schema comes from the SQLAlchemy models
(`create_all`); it does not exercise migrations because none exist yet. In CI
this runs as the separate `integration` job with service containers.

## Authorization and abuse controls

**Roles.** The role is stored on the user, copied into the dashboard token at
login, and enforced per endpoint (`app/api/deps.py`):

| Endpoint | admin | agent | viewer |
| --- | --- | --- | --- |
| List documents, list/read conversations | yes | yes | yes |
| Upload / delete documents | yes | no (403) | no (403) |
| Mint a chat ticket | yes | yes | no (403) |

`/auth/register` always creates an `admin`. There is no endpoint yet to add
`agent` or `viewer` users, so those rows are created directly in the database.
A role change takes effect at the user's next login: tokens last 24 h and are not
revoked.

**Secrets and accounts.**
- `ENVIRONMENT=production` refuses to start unless `JWT_SECRET` is at least 32
  characters, not a placeholder (for example `change_me...`) and not degenerate
  (for example one repeated character). Development stays permissive.
- Passwords must be 12 characters to 72 **bytes** (bcrypt's real limit; longer
  passwords were previously truncated silently). Emails are stored lowercase and
  matched case-insensitively. Login does one bcrypt check even for an unknown
  email, so timing does not reveal which emails exist. A duplicate email, including a lost
  race between two simultaneous registrations, returns 409 and leaves no orphan business.
- bcrypt runs in a worker thread, so a login no longer blocks the event loop.

**Rate limits** (Redis, fixed window; `RATE_LIMIT_*` variables; 429 with `Retry-After`):

| Action | Default | Keyed by |
| --- | --- | --- |
| Login | 10 / minute and 10 / 15 minutes | client IP; email |
| Register | 5 / hour | client IP |
| Mint chat ticket | 30 / minute | user |
| Chat messages | 20 / minute | session, checked first (WebSocket `rate_limited` frame with `retry_after`) |
| Chat messages | 60 / minute | tenant, checked second |

The session budget is checked before the tenant budget, so one flooding session is
stopped without spending the business's allowance and locking out its other customers.

**Pagination.** `GET /api/v1/conversations` and `GET /api/v1/documents` take `limit`
(1-500, default 100) and `offset`, return newest first, and send the total in the
`X-Total-Count` header; the body is still a plain JSON array. Messages inside one
conversation (`GET /api/v1/conversations/{id}`) take `limit` (1-2000, default 500) and
`offset`, and the response carries `message_count`. Out-of-range values get a 422.

Keys hash the IP or email, so Redis holds no raw personal data. Set
`REGISTRATION_ENABLED=false` to turn off self-service signup (the default is on
so local development works; production logs a warning while it is on).

**Known limits.**
- **Fails open:** if Redis is down the limiter allows requests (logged as
  `rate_limit_unavailable`) rather than locking every customer out.
- **Fixed windows** allow up to twice the limit across a window boundary.
- **Per-email limit:** an attacker can slow a victim's logins for up to 15
  minutes; there is no permanent lockout.
- **Behind a proxy**, run uvicorn with `--proxy-headers` and `--forwarded-allow-ips`
  so the limiter sees the real client IP; otherwise everyone shares one limit.
- Email lookup uses `lower(email)`, which the existing index does not cover.
  Fine at this scale; the proper fix is a lowercase unique index via a migration.
- Registration still reveals whether an email exists (inherent without email
  verification); the register limit slows enumeration.

## Authentication model

Two credentials, deliberately not interchangeable (`app/core/security.py`):

| Credential | Issued by | Lifetime | Authorizes | Rejected by |
| --- | --- | --- | --- | --- |
| Dashboard token (`scope=dashboard`) | `POST /api/v1/auth/login`, `/register` | 24 h | All REST endpoints (documents, conversations, ticket minting) | The WebSocket |
| Chat ticket (`scope=chat`) | `POST /api/v1/auth/chat-ticket` (needs a dashboard token) | `CHAT_TICKET_TTL_SECONDS`, default 60 s (allowed range 5-900) | One WebSocket session | All REST endpoints |

A ticket is bound to one tenant (`business_id`) and one server-generated,
unguessable session id (`sid`). The WebSocket path must be exactly
`/api/v1/voice/stream/{sid}`; any other session id is rejected. The tenant is
always taken from the verified token, never from anything the client sends.
Every token must carry `exp`, `iat`, `sub`, `business_id` and `scope`.

**Chat flow** (the customer-facing client never sees a dashboard token):

1. Your backend logs in and calls `POST /api/v1/auth/chat-ticket` with the
   dashboard token. The response has `ticket`, `session_id`, `expires_in`, `ws_path`.
2. It hands the ticket to the customer's client, which connects to
   `ws://<host>{ws_path}?token=<ticket>` and sends `{"text": "mera order kaha hai?"}`.

`scripts/test_chat.py` performs exactly this (password from `SHRIVOX_PASSWORD`
or a prompt). Seed-script login: `admin@example.com` / `demo_password_123`.

**Upgrade note:** tokens issued before this change have no `scope` claim and
are rejected; users simply log in again.

**Known limits of the ticket design**

- Browsers cannot set headers on a WebSocket handshake, so the ticket is sent
  in the URL query string, which web servers, proxies and load balancers may
  log. uvicorn's own logs are redacted (`app/core/log_redaction.py`), but
  infrastructure in front of the app is not covered: redact or disable
  `token=` in its logs too. The short lifetime and single-session scope are the
  real mitigation.
- A ticket is not single-use: it can be replayed on its session until it
  expires. Expiry is checked when the connection opens; an already-open
  connection is not cut off when the ticket expires.
- Ticket minting is rate limited per user (see "Authorization and abuse controls").

## Failure handling

Every external dependency can fail; each failure has a defined, tested outcome
(`tests/integration/test_failure_matrix.py`). Errors never expose exception
text, SQL or customer content: clients get fixed messages from one catalogue
(`app/core/failures.py`) and logs record only the error *type*.

**Chat WebSocket.** The server answers each client message with exactly one frame:
a reply, `{"type": "response", "response", "intent", "escalate", "safety_flag", "latency_ms"}`,
or an error, `{"type": "error", "code", "message", "retryable"}`. An error frame does **not**
close the connection. `retryable` means "the same message may succeed if sent again".

| What fails | Client sees | Postgres | Redis |
| --- | --- | --- | --- |
| Gemini timeout / 429 / 5xx / network (after bounded retries) | `llm_unavailable`, retryable | nothing written | nothing written |
| Gemini 4xx (bad request, bad key; never retried) | `llm_rejected` | nothing written | nothing written |
| Gemini returns no text | `llm_empty_response`, retryable | nothing written | nothing written |
| Gemini returns no intent text | normal reply; intent falls back to `general_query` | turn saved | turn saved |
| Query embedding fails | `embedding_unavailable` (no answer is invented without knowledge-base search) | nothing written | nothing written |
| Postgres fails while retrieving or saving | `database_unavailable`, retryable; the reply is withheld | rolled back; session stays usable | **not** written |
| Postgres unreachable when the session opens | `database_unavailable`, then close code 1011 | none | none |
| Redis unreachable (or timing out after `REDIS_TIMEOUT_SECONDS`) | normal reply with `"degraded": ["session_memory"]` | turn saved | skipped; the next turns lack earlier context |
| Turn exceeds `TURN_TIMEOUT_SECONDS` | `turn_timeout`, retryable | nothing written | nothing written |
| Unexpected server error | `internal_error` | nothing written | nothing written |
| Bad frame (not JSON, not an object, binary) | `invalid_frame` | nothing written | nothing written |
| Missing / empty / non-string `text` | `invalid_message` | nothing written | nothing written |
| `text` longer than `MAX_MESSAGE_CHARS` | `message_too_long` | nothing written | nothing written |

*Ordering:* Postgres is written first and a reply is delivered only after it is
saved, so a delivered reply is always persisted. Redis is working memory, written
second; a customer message and its reply are stored in one write.

*Call lifecycle:* a call is `active` while connected, `escalated` as soon as an
escalating turn is saved, and on disconnect becomes `completed` (or stays
`escalated`) with `ended_at` set. Reconnecting to a finished call makes it `active` again.

**Gemini calls** (chat and embeddings) share one policy (`app/core/upstream.py`):
a per-attempt timeout, and at most `UPSTREAM_MAX_ATTEMPTS` attempts with
exponential backoff for *transient* errors only (timeout, network, HTTP 408/429/5xx).
Other 4xx errors are never retried.

**Document upload** embeds everything *first*, then writes the document and all its
chunks in one transaction, so a failure leaves nothing behind (no half-ingested
document). Embedding outage: `503 embedding_unavailable` (with `Retry-After`);
embedding rejected: `502`; database outage: `503 database_unavailable`; unexpected
database rejection: `500 internal_error`. Invalid uploads (empty or whitespace-only
text, `text` over `MAX_DOCUMENT_CHARS`, filename empty or over 255 characters) get
`422` before any Gemini call. The same JSON shape, `{"detail", "code"}`, is used by
every REST endpoint when the database or Gemini fails.

| Variable | Default | Meaning |
| --- | --- | --- |
| `UPSTREAM_TIMEOUT_SECONDS` | 15 | Timeout per Gemini attempt |
| `UPSTREAM_MAX_ATTEMPTS` | 3 | Attempts for transient errors (1-5) |
| `UPSTREAM_RETRY_BACKOFF_SECONDS` | 0.5 | Exponential backoff multiplier |
| `TURN_TIMEOUT_SECONDS` | 45 | Ceiling for one whole chat turn |
| `MAX_MESSAGE_CHARS` | 2000 | Longest accepted chat message |
| `MAX_DOCUMENT_CHARS` | 200000 | Longest accepted document |
| `REDIS_TIMEOUT_SECONDS` | 2.0 | Redis connect/read timeout |

**Known limits**

- Request-body and WebSocket frame sizes are only checked *after* the server has
  read them. Enforce hard limits at the reverse proxy / `uvicorn --ws-max-size`.
- If the server is shut down abruptly (the handler task is cancelled), a call row
  can remain `active`; nothing reaps stale calls yet.
- There are no idempotency keys: a client retry after a lost response creates a
  second turn (chat) or a second document (upload).
- Embedding batches use the SDK's list input; a wrong-sized response falls back to
  one call per text (logged as `embed_batch_count_mismatch`). Batch behaviour has
  not been verified against the live API.
- The Redis timeout setting is applied, but only the connection-refused path is covered by a test, not an actual slow Redis.

## Evaluation

```
python -m app.evaluation.run_eval
```

A deterministic eval harness (no LLM-as-judge) against a labeled
Hinglish + English test set. It measures:

- **Intent accuracy** — does routing work
- **Retrieval precision/recall** — is RAG pulling the right *category* of doc
- **Avg end-to-end latency** — is this fast enough for a live call
- **Safety flag rate** — how often the output guardrail intervened

**Status of `run_eval`: harness is built; no baseline published yet.** The measured
results below come from the separate RAG evaluation (`rag_eval`).

### RAG evaluation results

```
python -m app.evaluation.rag_eval --corpus realistic --pause 2 --llm-retries 3
```

Live run against Gemini on `main` (6d3b0d4), 2026-11-20, 61 questions over the 52-entry
realistic corpus: 52 questions have a known relevant document and are scored for retrieval;
the other 9 are questions the corpus does not cover.

| Measure | Result |
| --- | --- |
| Questions run / errors | 61 / 0 |
| Recall@3, Hit Rate@3, MRR (52 retrieval-scored questions) | 1.00, 1.00, 1.00 |
| Answer relevance, judged by Gemini (1-5) | 4.98 |
| Answer faithfulness, judged by Gemini (1-5) | 4.98 |
| Reply language matches the question (heuristic detector) | 1.00 |
| Raw answers flagged by the output guard | 0 of 61 |
| Retrieval latency, includes the query-embedding call | p50 532 ms, p95 877 ms |

One answer scored 4 on both judged measures (h-52, an assembled sofa: the judge marked down an
offer to escalate to a human); the other 60 scored 5. The offline similarity-floor sweep on this
run: at floors up to 0.50 recall stays 1.00 while all 9 uncovered questions still retrieve some
context; at 0.60 recall stays 1.00 and that drops to 0.67; at 0.70 recall falls to 0.85. The
production floor (0.25) has not been changed.

**Limits.** The corpus and questions are self-written, so these numbers compare versions of
this system; they are not an accuracy claim for production traffic. The judge is an LLM and its
prompt was tightened after the first live report. Each question was run once, and the answers are
the model's raw output, before the output guard's regenerate-and-hand-off loop. Latency is
retrieval only, not a whole chat turn.

## Observability

`GET /metrics` exposes Prometheus metrics (HTTP by route *template*, LLM calls,
latency, tokens, truncated answers, guardrail interventions). No customer text
is ever a label. Logs are JSON and honour `LOG_LEVEL`. Keep `/metrics` off the
public ingress.

## Retrieval at scale (HNSW) and the realistic eval corpus

* **Index.** At startup (and in `scripts/init_db.py`) `app/db/vector_index.py` builds an HNSW index on
  `knowledge_chunks.embedding` **only when pgvector >= 0.8**, and retrieval then enables
  `hnsw.iterative_scan`. Reason: pgvector applies the tenant `WHERE business_id = ...` *after* the index
  scan, so without iterative scans a small tenant can silently get fewer than k results when other tenants'
  vectors crowd the candidates (reproduced and tested in `tests/integration/test_hnsw_tenant_recall.py`).
  On older pgvector no index is created and searches stay exact. The build runs in the background
  (about 2 minutes for 40k chunks at 1536 dims) and uses `CREATE INDEX CONCURRENTLY`, so writes are not blocked;
  until it finishes, searches use the exact path.
* **Benchmark** (`scripts/bench_vector_search.py`, synthetic vectors, scratch database; saved result:
  `docs/bench_vector_search_40k.json`). 40,000 chunks, 75 tenants, pgvector 0.8.0: the largest tenant
  (20k chunks) went from ~110 ms to ~4.5 ms p50 with recall@3 = 1.0; small (50) and medium (500) tenants
  were already fast (the btree on `business_id` handles them) and stayed the same. So the index pays off for
  *large tenants*, not for small ones. No query returned fewer than k rows in either state.
* **Realistic eval.** `python -m app.evaluation.rag_eval --corpus realistic` seeds a ~52-entry corpus
  (near-duplicate policies, Hinglish, typos, follow-ups, questions the KB does not cover) into a **throwaway
  tenant that is deleted afterwards**, reports Recall@K / Hit Rate@K / MRR by question kind, retrieval latency,
  and an offline **similarity-floor sweep** (missing-context rate vs irrelevant-context rate per threshold).
  It also runs the index set-up first, so on a database without the index it will build one (as startup would).
  Needs Postgres and `GOOGLE_API_KEY` in `.env` (embeds ~58 chunks). Self-written data: compare versions, don't
  quote it as an accuracy figure.

## Answer-generation hardening (from the first live RAG eval)

* **System prompt** (`app/prompts/templates.py`): reply in the language of the customer's *latest* message
  (English gets plain English; the old "don't force pure English" line is gone); never say the company does not
  offer/accept something unless the context says so; never assume a category (a city is not a "metro") the
  customer and context did not state. `tests/test_prompt_contract.py` pins these rules. A test cannot prove the
  model obeys them; only a rerun of the live eval can.
* **Output guard**: `unsupported_negative` flag (in `app/agents/guardrails.py`) catches "we don't accept X" /
  "X is not available" when X is absent from the retrieved context; the answer is regenerated once with a
  targeted instruction, then handed to a human if it persists. Deliberately conservative (English and romanised
  Hinglish shapes seen in real output).
* **Gemini rate limits** (`app/core/upstream.py`): a 429's server-stated retry delay (RetryInfo / Retry-After) is
  honoured instead of guessed; a delay above `UPSTREAM_MAX_RETRY_WAIT_SECONDS` (default 10) fails fast, and the
  HTTP 503 carries the real `Retry-After`.
* **Eval** now records failure stage/reason (never content), retries LLM stages (`--llm-retries`, paced with
  `--pause`), passes history to follow-up answers, and adds deterministic per-answer checks: language match and
  what the output guard would flag.

## Scoped figures (the Pune case)

A live eval question, "Will my order reach Pune in 5 days?", showed that every number in an
answer can appear in the retrieved text and still be wrong for the customer. The policy said
standard delivery takes 4-7 business days to six listed metro cities and 7-10 business days to
all other locations; retrieval ranked it first, but the model gave Pune the 4-7 figure, and the
numeric guard passed it because 4-7 is in the context.

`app/agents/scope.py` finds the supported sentence pattern: one delivery figure for a listed
group of cities and another figure for all other locations. Before generation it adds a note to
the prompt saying which figure applies to this customer; after generation it flags an answer that
gives the listed-city figure when the customer named a city outside the list (`scope_violation`:
one targeted regeneration, then a human hand-off). Only the customer's own words count as
naming a place, never the agent's earlier turns or the answer itself. In the live run above, Pune and Gurgaon got 7-10 days,
"Bangalore" was recognised as a listed metro (4-7), and Mumbai got 4-7. A question naming no
city asked the customer to specify their location before confirming the delivery timeline.

**Limits.** It handles only that one sentence shape (not the reverse word order, and not a rule
split across two retrieved passages); the city-alias table is small (Bangalore/Bengaluru and a
few others) plus typo tolerance, so an unrecognised alias leads to a retry or hand-off rather
than a confident wrong figure; each case was run once. See `docs/AUDIT.md`.

## Guardrail regression eval

```
python -m app.evaluation.guardrail_eval
```

Offline (no API key). The labelled cases are self-written, so the score is a
regression signal, not a production accuracy estimate; see `docs/AUDIT.md`.

## Testing

```
pytest tests/ -v
```

All external calls (Gemini, Postgres, Redis) are mocked in unit tests
— they check the graph's *branching logic* (escalation short-circuits
the LLM, safety-flag triggers exactly one retry), not LLM output
quality.

## Known gaps / honest scope

- Voice (streaming ASR/TTS) and Twilio telephony are not yet wired in
  — the WebSocket contract is designed to extend to them without
  re-architecting, but they're not built yet.
- `run_eval` (intent accuracy, category-level retrieval, end-to-end latency) has not been
  published as a baseline; the measured results are the RAG evaluation above, on a
  self-written set, with retrieval latency only and no load test.
- Intent classifier doesn't yet distinguish gratitude/confirmation
  from greeting/general_query (e.g. "thanks" and "ok got it" both
  route generically) — flagged as a near-term fix.
- Order-tracking / CRM lookups are not wired to any real backend yet
  — the agent can detect the intent and escalate, but can't fetch
  live order data.
- Admin dashboard is not yet built; the `calls`/`messages` schema is
  designed to support one (see `app/db/models.py`).
- Guardrails are pattern-based, not a second LLM judge call, as a
  deliberate latency/cost tradeoff — documented in
  `app/agents/guardrails.py`.
- No Alembic migrations yet (`scripts/init_db.py` uses `create_all`)
  — reasonable for a schema this new; Alembic is already a dependency
  for when it's worth the ceremony.
- Document ingestion accepts raw text via JSON, not file upload +
  PDF/DOCX parsing yet.
