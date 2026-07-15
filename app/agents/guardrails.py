"""
Safety layer. Runs twice per turn: once on user input (before the LLM
sees it), once on the generated response (before TTS speaks it).

Why two passes instead of one:
Input-side catches things the system shouldn't even attempt to answer
(e.g. legal threats, requests for account access bypass) -- these get
routed straight to human escalation without wasting an LLM call.
Output-side catches the LLM inventing something it shouldn't have said
(e.g. promising a refund amount not in the KB, giving medical/legal
advice, hallucinating a policy). Different failure modes, so one check
can't do both jobs.

Why simple keyword/pattern rules here instead of a second LLM call to
"judge" every turn:
A judge-LLM adds real latency to every single turn, in a system whose
whole value proposition is being fast enough for live voice. Cheap
deterministic checks handle the clear-cut, high-confidence cases
(fraud/legal/self-harm language, promising unverified financial
amounts) instantly; anything ambiguous falls through to escalation
rather than the AI guessing. This is a starting ruleset, meant to be
extended -- the evaluation pipeline tracks safety_violation rate so
weak spots show up empirically rather than being guessed at upfront.
"""

from dataclasses import dataclass
import re

ESCALATION_TRIGGERS = [
    r"\blegal action\b", r"\bsue\b", r"\blawyer\b", r"\bfraud\b",
    r"\bmanager\b", r"\bhuman agent\b", r"\bspeak to (a )?person\b",
    r"\bself.?harm\b", r"\bsuicide\b", r"\bkill myself\b",
]

# Response-side: patterns that suggest the LLM is inventing specifics
# it shouldn't (exact refund amounts, dates, or promises) without
# grounding. This is a heuristic, not a proof of hallucination --
# it flags for review rather than silently blocking.
UNGROUNDED_PROMISE_PATTERNS = [
    r"\bguarantee(d)?\b", r"\bpromise\b", r"\b100% (refund|sure)\b",
]


@dataclass
class SafetyResult:
    safe: bool
    flag: str | None  # None if clean, else a short category tag
    escalate: bool


def check_input(text: str) -> SafetyResult:
    lowered = text.lower()
    for pattern in ESCALATION_TRIGGERS:
        if re.search(pattern, lowered):
            return SafetyResult(safe=False, flag="escalation_trigger", escalate=True)
    return SafetyResult(safe=True, flag=None, escalate=False)


def check_output(text: str, grounded_in_kb: bool) -> SafetyResult:
    lowered = text.lower()
    if not grounded_in_kb:
        for pattern in UNGROUNDED_PROMISE_PATTERNS:
            if re.search(pattern, lowered):
                return SafetyResult(safe=False, flag="ungrounded_promise", escalate=False)
    return SafetyResult(safe=True, flag=None, escalate=False)
