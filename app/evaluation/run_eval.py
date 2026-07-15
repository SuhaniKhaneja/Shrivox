"""
Evaluation pipeline. Requires the demo business to be seeded first:

    python scripts/seed_demo_business.py
    python -m app.evaluation.run_eval

Why a labeled test set instead of eyeballing conversations:
"The bot seemed to work" is not a metric, and it doesn't tell you
whether a prompt change made things better or worse. A fixed eval set
gives a repeatable number to compare across changes -- this is the
same reason ML models get held-out test sets instead of "looks right
to me." Every metric here maps directly to a real production concern:

- intent_accuracy: does the routing even work
- retrieval_precision/recall: is RAG pulling the right *category* of
  doc (chunk ids are now DB-generated UUIDs, so we check the
  chunk's category metadata rather than an exact id -- if this is
  low, no amount of prompt tuning fixes the answers, the fix is in
  retrieval, not the prompt)
- avg_latency_ms: is this fast enough to feel like a live phone call
- safety_violation_rate: how often the output guardrail had to fire

This is intentionally a fast, deterministic eval (no LLM-as-judge)
for Phase 1/2. An LLM-judge pass for subjective quality (tone,
helpfulness -- faithfulness/groundedness per the system design doc)
is a natural next addition once there's a numeric baseline.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path

from sqlalchemy import select

from app.agents.graph import shrivox_agent
from app.db.session import AsyncSessionLocal
from app.db.models import Business, KnowledgeChunk

EVAL_PATH = Path(__file__).parent / "eval_dataset.json"
DEMO_BUSINESS_NAME = "Demo E-Commerce Co"  # matches scripts/seed_demo_business.py


async def resolve_demo_business_id() -> uuid.UUID:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Business).where(Business.name == DEMO_BUSINESS_NAME))
        business = result.scalar_one_or_none()
        if not business:
            raise RuntimeError(
                f"No business named '{DEMO_BUSINESS_NAME}' found -- "
                "run `python scripts/seed_demo_business.py` first."
            )
        return business.id


async def categories_for_chunk_ids(chunk_ids: list[str]) -> set[str]:
    if not chunk_ids:
        return set()
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.id.in_([uuid.UUID(c) for c in chunk_ids]))
        )
        chunks = result.scalars().all()
        return {c.chunk_metadata.get("category") for c in chunks if c.chunk_metadata.get("category")}


async def run_single(case: dict, business_id: uuid.UUID) -> dict:
    start = time.perf_counter()
    result = await shrivox_agent.ainvoke({
        "session_id": f"eval-{case['id']}",
        "business_id": business_id,
        "user_message": case["user_message"],
        "conversation_history": [],
        "intent": None,
        "retrieved_docs": [],
        "retrieved_doc_ids": [],
        "response_text": None,
        "safety_flag": None,
        "escalate": False,
        "latency_ms": 0,
        "token_count": 0,
    })
    latency_ms = int((time.perf_counter() - start) * 1000)

    intent_correct = result["intent"] == case["expected_intent"]

    expected_categories = set(case["expected_categories"])
    retrieved_categories = await categories_for_chunk_ids(result["retrieved_doc_ids"])

    if expected_categories:
        precision = len(expected_categories & retrieved_categories) / max(len(retrieved_categories), 1)
        recall = len(expected_categories & retrieved_categories) / len(expected_categories)
    else:
        precision, recall = 1.0, 1.0  # no retrieval expected (e.g. escalation cases)

    return {
        "id": case["id"],
        "intent_correct": intent_correct,
        "predicted_intent": result["intent"],
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "latency_ms": latency_ms,
        "safety_flag": result["safety_flag"],
        "escalated": result["escalate"],
        "response": result["response_text"],
    }


async def main():
    business_id = await resolve_demo_business_id()
    cases = json.loads(EVAL_PATH.read_text())
    results = [await run_single(c, business_id) for c in cases]

    n = len(results)
    intent_accuracy = sum(r["intent_correct"] for r in results) / n
    avg_precision = sum(r["retrieval_precision"] for r in results) / n
    avg_recall = sum(r["retrieval_recall"] for r in results) / n
    avg_latency = sum(r["latency_ms"] for r in results) / n
    safety_violation_rate = sum(1 for r in results if r["safety_flag"]) / n

    print("\n=== Shrivox AI Evaluation Report ===")
    print(f"Test cases:              {n}")
    print(f"Intent accuracy:         {intent_accuracy:.1%}")
    print(f"Retrieval precision:     {avg_precision:.1%}")
    print(f"Retrieval recall:        {avg_recall:.1%}")
    print(f"Avg end-to-end latency:  {avg_latency:.0f} ms")
    print(f"Safety flag rate:        {safety_violation_rate:.1%}")
    print("\nPer-case detail:")
    for r in results:
        status = "OK" if r["intent_correct"] else "MISS"
        print(f"  [{status}] {r['id']}: intent={r['predicted_intent']} "
              f"precision={r['retrieval_precision']:.2f} latency={r['latency_ms']}ms "
              f"flag={r['safety_flag']}")

    Path("eval_report.json").write_text(json.dumps({
        "summary": {
            "intent_accuracy": intent_accuracy,
            "retrieval_precision": avg_precision,
            "retrieval_recall": avg_recall,
            "avg_latency_ms": avg_latency,
            "safety_violation_rate": safety_violation_rate,
        },
        "cases": results,
    }, indent=2))
    print("\nFull report written to eval_report.json")


if __name__ == "__main__":
    asyncio.run(main())
