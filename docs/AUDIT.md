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


## Addendum: retrieval at scale (HNSW) and realistic RAG eval

Verified: pgvector 0.8.0 locally (iterative scan available); CI image is `pgvector/pgvector:pg16`
(unpinned tag, currently >= 0.8). **Correction to an earlier claim of mine:** I said search cost grows with the
whole platform. Measured, it grows with the *tenant's own* size: the planner already uses the btree on
`business_id`, so small tenants were fast without any vector index. The index matters for large tenants.

| tenant size (40k chunks, 75 tenants) | state | recall@3 | p50 ms | p95 ms |
|---|---|---|---|---|
| small (50) | no index | 1.000 | 4.09 | 4.82 |
| small (50) | HNSW + iterative | 1.000 | 3.77 | 4.04 |
| medium (500) | no index | 1.000 | 5.80 | 6.59 |
| medium (500) | HNSW + iterative | 1.000 | 5.26 | 7.70 |
| large (20,000) | no index | 1.000 | 109.84 | n=3 |
| large (20,000) | HNSW + iterative | 1.000 | 4.55 | n=3 |

Limits of this benchmark: synthetic vectors; database time only (no embedding call); the "large" row is ONE
tenant sampled 3 times, so treat ~24x as indicative, not a distribution. Index build: 138 s for 40k chunks.

Tests that prove the mechanism (each checked by mutation): without iterative scan a small tenant is starved
(fewer than k rows) on the HNSW plan; with it the rows are complete, identical to an exact scan and tenant-scoped;
removing the tenant filter, switching iterative scan off, awaiting the build at startup, or skipping the
throwaway-tenant cleanup each fails a test.

Realistic corpus: 52 entries / 58 chunks / 52 questions (kinds: direct, paraphrase, hinglish, typo, follow-up,
long-doc, unanswerable). I checked every label against the corpus text by hand; two assertions in the first
draft of the tests had never been run and were wrong (chunk count; a word-based proxy for "has a number"), so the
corpus was extended and the second assertion made digit-based.

NOT validated: retrieval quality with real Gemini embeddings on this corpus (needs your key); the threshold sweep
result (same); behaviour of the planner on your production data distribution; the unpinned CI image version.
At ~58 chunks the planner will choose an exact scan, so the realistic eval measures embedding/ranking quality,
not HNSW; HNSW is covered by the integration tests and the benchmark.


## Addendum: answer-generation hardening (after the first live Gemini RAG eval)

Live eval (52 questions, user-run): retrieval saturated (Recall@3 = Hit@3 = MRR = 1.0); answer relevance and
faithfulness 4.85/5 over the 48 judged. Findings and what was done:

| Finding (report) | Change | Verified how |
|---|---|---|
| h-14: Jaipur called a "metro" (context lists the metros; 1/1 from the judge). The numeric guard cannot catch it: 4-7 days IS in the context, it is attributed to the wrong entity. | Prompt rule: no assumed categories; unlisted city -> general rule. Judge prompt penalises it. New `category` eval cases (h-53..h-55). | Prompt-contract test only. **No deterministic guard exists for this; effect unproven until a live rerun.** |
| h-47 / h-49: "we don't accept Bitcoin", "gift wrapping not available" with no KB support (2 of 6 unanswerable) | Prompt rule + `unsupported_negative` output guard + targeted regeneration; 3 more unanswerable cases (h-56..h-58) | Guard flags exactly h-47 and h-49 on the 49 real answers, 0 false positives; BUT I fixed two false positives (h-36, h-41) on that same data, so this is calibration, not a held-out result. |
| 24 of 44 English-question answers were Hinglish (55%), judged 5/5 regardless; the old prompt line "don't force pure English" biased it | Prompt rewritten; deterministic language check added to the eval (the judge does not score language) | Detector baseline measured on the report; whether the new prompt fixes it needs a live rerun. |
| 4 of 52 calls failed `UpstreamUnavailable` (cause not recorded); both follow-up cases among them | Retry honours the server's retry delay, fails fast beyond a cap, reports the real Retry-After; eval records stage/reason, retries, paces (`--pause`), and now gives follow-ups their history | Unit tests + mutation checks. **The cause of those 4 failures remains unknown** (rate limit is the likely candidate, not confirmed). |

Not changed, by decision: retrieval and the 0.25 similarity floor. In the report no floor separates answerable from
unanswerable questions (relevant top scores 0.646-0.821; unanswerable 0.570-0.680, overlapping).

## Addendum: scoped figures (live eval case h-53, "Will my order reach Pune in 5 days?")

**What happened.** v4 live eval: 58/58 answered, retrieval correct (`ship-standard` #1), but the answer said 4-7 business
days. The corpus lists six metro cities for 4-7 days and gives 7-10 days to "all other locations".

**Why the guards missed it (verified deterministically, no Gemini needed).** `check_output` asks "is this figure in the
context?". `4-7 business days` is, so the answer passed. The prompt rule ("if the customer's city is not listed, use the
general rule") left a set-membership test to the model. That the model was pulled toward the nearer range because the
customer wrote "5 days" is a hypothesis; the v4 report's raw answer was not available to me.

**Fix (generic, `app/agents/scope.py`, no extra LLM call).** Finds "<figure> ... (list) ... <other figure> ... all other ..."
sentences in the retrieved passages (cities, products, payment methods alike). Then:
1. before generation, adds a "Scope notes" block to the prompt saying which figure applies to this customer;
2. after generation, `check_scope` flags an answer that gives the listed figure (without the other one) when the
   customer named nothing on the list -> flag `scope_violation`, one targeted regenerate, then human hand-off.
Only the customer's own words count as naming a place; the agent's earlier turns and the answer itself never do.

**Limits.** Alias table is small (Bangalore/Bombay/Madras/Gurgaon...) plus typo tolerance, so an unrecognised alias for a
listed city leads to a retry/hand-off, never a confident wrong figure. Only the "listed group + all other" shape is parsed
(not reversed word order, not a rule split across chunks). Whether the notes make Gemini answer h-53 correctly is **not
verified here**: needs a live rerun (`h-53`, plus new h-59 Bangalore alias, h-60 no city named, h-61 Gurgaon).
