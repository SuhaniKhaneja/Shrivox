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
[detect_intent] --escalation trigger--> [generate_response: transfer message] --> END
   |  (no trigger)
   v
[retrieve_context]  (pgvector, scoped to business_id, top-3, score-thresholded)
   |
   v
[generate_response]  (Gemini 2.5 Flash, grounded in retrieved context)
   |
   v
[safety_check_output] --ungrounded promise--> retry once with stricter prompt
   |
   v
Response -> Redis (session memory) + Postgres (durable record) -> client
```

Built as a **LangGraph StateGraph**, not a linear function, because the
branches are genuinely conditional: an escalation-triggering message
must skip retrieval and generation entirely (latency + cost), and an
output that trips the safety check needs a retry path, not a silent
pass-through. See `app/agents/graph.py` for the reasoning behind each
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
  api/
    deps.py               Shared auth dependency (JWT -> CurrentUser)
    auth.py                Business registration + login
    documents.py            Knowledge-base upload/list/delete
    conversations.py         REST call history + WebSocket chat endpoint
  agents/
    graph.py               LangGraph state machine (the core agent)
    guardrails.py            Input/output safety checks
    llm_client.py             Gemini wrapper (retry + latency capture)
  retrieval/
    retriever.py             pgvector search + Google embeddings (RAG)
    chunking.py                Word-based chunking with overlap
  memory/
    session_memory.py          Redis-backed hot conversation state
  db/
    models.py                    Multi-tenant Postgres schema
    session.py                     Async session factory
  prompts/
    templates.py                    System prompt + intent/response templates
  evaluation/
    eval_dataset.json                 Labeled test cases (Hinglish + English)
    run_eval.py                        Evaluation harness (see below)
data/knowledge_base/        Sample Hinglish support KB (JSON)
scripts/
  init_db.py                    Creates the pgvector extension + tables
  seed_demo_business.py           Seeds a demo tenant + admin user + KB
tests/                       pytest suite, external calls mocked
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

Open `http://localhost:8000/docs` for interactive API docs. Log in
via `POST /api/v1/auth/login` with the credentials the seed script
printed, then connect to `ws://localhost:8000/api/v1/chat/stream/{session_id}?token=<access_token>`
and send `{"text": "mera order kaha hai?"}`.

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

**Status: harness is built, not yet run against the full labeled set.**
Running it and publishing a baseline is the next milestone before
moving to Phase 2.

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
- Evaluation harness has not been run against the full labeled set —
  no accuracy/latency numbers published yet.
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
