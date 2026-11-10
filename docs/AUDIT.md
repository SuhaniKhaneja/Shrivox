# Shrivox audit and improvement log

Method: read the whole codebase, ran it, wrote a failing test or probe for each suspected
defect *before* fixing it, then fixed and re-ran everything. Final state: **407 tests pass**
(317 unit + 90 integration against real Postgres/pgvector and Redis), ruff (correctness rules)
clean, unit coverage of `app/` 89%.

## What changed, in the order done

| # | Problem found | Change | Evidence |
|---|---|---|---|
| P1a | Session memory was read-modify-write on Redis: concurrent turns lost messages | Atomic `RPUSH+LTRIM+EXPIRE+LRANGE` in one MULTI/EXEC; key prefix bumped to `v2` (old keys are simply ignored) | real-Redis concurrency test |
| P1b | Two simultaneous first messages created duplicate `Call` rows | Postgres advisory transaction lock; existing duplicates heal to the earliest row | real-Postgres race test |
| P1c | Unbounded list endpoints; no per-session rate limit | Pagination (`limit/offset`, `X-Total-Count`); per-session limit checked before tenant limit | integration tests |
| P2a | Graph was a linear function with decorative edges | Real LangGraph: guard -> parallel classify+retrieve -> join -> triage; bounded safety loop; safe fallback; `token_count` now sums *all* LLM calls (this changes the stored number: it used to be last-writer-wins) | topology, concurrency and fallback tests |
| P2b | Follow-ups like "aur kitne din lagenge?" retrieved nothing | Second query borrowing the previous customer message, one batched embedding request, tenant filter on both searches | real-pgvector test |
| P3 | Guardrails were English-only keywords, and only checked the words "guarantee/promise" | See below | `tests/test_guardrails.py`, `guardrail_eval` |
| P4 | `LOG_LEVEL` unused, no metrics, silent answer truncation | `/metrics`, JSON logs honouring `LOG_LEVEL`, `finish_reason == MAX_TOKENS` detection | `tests/test_observability.py`, live smoke |
| P4b | CI had no lint/coverage/eval; Docker ran as root; scripts needed `PYTHONPATH`; stale model/embedding docs | fixed all | CI file, Dockerfile |

### Guardrails v2
Input rules: English, romanised Hinglish and Devanagari; Unicode NFKC normalisation (full-width,
spacing, case tricks); a separate `self_harm` flag that wins over other rules and gets a caring
reply rather than the routine hand-off line; hard negatives ("my manager told me...", "Sue",
"fraud-proof wallet") no longer trip it. Output check: besides unsupported promises, a numeric
grounding check on amounts, percentages and durations (normalised: 1 week = 7 days = 168 h,
"three" = 3, "30-day" = 30 days), supported by the KB *or* by what the customer already said.
An unsafe answer is regenerated once with an instruction targeted at the flag, checked again, and
replaced by a human hand-off if still unsafe.

**Honest numbers.** The labelled set (98 input / 46 output cases) is self-written, so 1.00/1.00
is a regression signal, not an accuracy claim. The first held-out batch I wrote *after* the rules
scored worse than that; I fixed the rules and added those cases to the set, so they are no longer
held out. Two known misses are recorded in `guardrail_cases.json` under `known_gaps`
(e.g. "sab khatam kar dunga apni jaan": verb-before-noun word order).

## Could NOT be verified (no Gemini key in this environment)
- Real answer quality, intent-classifier accuracy and retrieval relevance. Every test fakes the model.
- Whether the real SDK returns `finish_reason` in the shapes my truncation check accepts (it handles
  enum, string and missing; not tested against a live response).
- That the numeric check's false-positive rate is acceptable on real model output. It errs toward
  re-generating / handing off; watch `shrivox_guard_events_total{flag="ungrounded_specifics"}`.
- The Docker build and healthcheck (no Docker daemon here); the GitHub Actions workflow itself has not run.
- Behaviour at real load: no load test was done.

## Deliberately not done
Tool-calling agent loop (needs a real order backend to be meaningful), streaming responses,
Alembic migrations, HNSW index (matters only at KB sizes I couldn't test), prompt-injection
hardening of retrieved documents, persisting eval runs (the `Evaluation` table requires a
`business_id`, which the offline guardrail eval has no meaning for). Truncated answers are
counted and logged but still delivered; the next step is to raise `max_tokens` or retry.
