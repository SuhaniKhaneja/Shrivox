"""
Safety layer. Runs twice per turn: once on user input (before the LLM
sees it), once on the generated response (before TTS speaks it).

Why two passes instead of one:
Input-side catches things the system shouldn't even attempt to answer
(e.g. legal threats, requests for a human, self-harm language) -- these get
routed straight to human escalation without wasting an LLM call.
Output-side catches the LLM inventing something it shouldn't have said
(a refund amount, a delivery window, a discount that is not in the KB, or a
promise it cannot back). Different failure modes, so one check can't do both
jobs.

Why deterministic rules instead of a second LLM call to "judge" every turn:
A judge-LLM adds real latency to every single turn, in a system whose value
proposition is being fast enough for live voice. Cheap deterministic checks
handle the clear-cut, high-confidence cases instantly, and anything ambiguous
falls through to escalation rather than the AI guessing.

What changed from the first version (see docs/AUDIT.md):
  * Input rules were English-only in a Hinglish product. They now cover
    English, romanised Hinglish and Devanagari ("kisi insaan se baat karni
    hai", "consumer court jaunga", "यह धोखा है").
  * Several rules fired on innocent text: "my manager told me...", a customer
    named Sue, "fraud-proof wallets", "lawyer-grade briefcase". Rules now
    require intent-bearing context (a verb, a determiner, a non-possessive).
  * Self-harm language is its own category, because it needs a different reply
    from "let me connect you with an agent about this".
  * The output check only ever looked for the words "guarantee"/"promise".
    The costliest hallucination in support -- an invented amount, percentage
    or number of days -- passed untouched, and a grounded answer was never
    checked at all. Specifics (money, percentages, durations) are now
    extracted from the answer and must be supported by the retrieved context
    or by what the customer / agent already said.

Limits worth knowing: these are heuristics tuned on a small self-written set
(app/evaluation/guardrail_cases.json). The numeric check verifies that a figure
APPEARS in the allowed text, not that the sentence means the same thing, and it
only sees digits and a short list of number words. Treat its precision as
something to measure on real transcripts, not as a given.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class SafetyResult:
    safe: bool
    flag: str | None  # None if clean, else a short category tag stored with the message
    escalate: bool
    category: str | None = None  # finer rule family, for logs/metrics (never the matched text)
    detail: tuple[str, ...] = ()  # e.g. ("money:5000",): what was unsupported, for logs


# ----------------------------------------------------------------- input rules

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


# Not preceded by a possessive: "my manager" is the customer's own boss, not a request.
_NOT_POSSESSIVE = r"(?<!my )(?<!our )(?<!his )(?<!her )(?<!their )(?<!mere )(?<!mera )"
_HUMAN_NOUN = (r"(?:person|human|agent|representative|rep|executive|someone|somebody|supervisor|"
               r"manager|staff|team member|team|operator|officer)")
_KARNA = r"(?:kar|krn|krw|karw)\w*"  # karo / karni / karna / karwao / karao ... (a request, not "baat hui")

_RULES: list[tuple[str, str, re.Pattern]] = []


def _rule(category: str, flag: str, *patterns: str) -> None:
    for pattern in patterns:
        _RULES.append((category, flag, re.compile(pattern)))


# Self-harm first: when several rules match, the most serious category wins.
_rule("self_harm", "self_harm",
      r"\bself[\s-]?harm\b", r"\bsuicid(?:e|al)\b", r"\bkill(?:ing)? myself\b",
      r"\bend(?:ing)? my (?:own )?life\b", r"\b(?:want|wanna|going) to die\b(?! (?:laughing|of|from|for))",
      r"\bdon'?t want to (?:live|be alive)\b(?! with (?:this|that|the|it|these|those)\b)",
      r"\bhurt(?:ing)? myself\b", r"\bend it all\b", r"\bapni jaan (?:de|le|khatam)",
      r"\bkhudkushi\b", r"\baatmahatya\b", r"\bmujhe (?:ab )?marna\b",
      r"\bmar jaunga\b", r"\bmar jaungi\b", r"\bjaan de (?:dunga|dungi)\b",
      r"आत्महत्या", r"खुदकुशी")

_rule("human_request", "escalation_trigger",
      rf"\b(?:speak|talk|chat)\s+(?:to|with)\s+(?:a |an |the |your |one of your )?"
      rf"(?:(?:real|live|actual|human|senior|customer care|customer service|support) )*{_HUMAN_NOUN}\b",
      r"\b(?:real|live|actual|human)\s+(?:person|human|agent|being|representative|executive)\b",
      r"\bcustomer\s+(?:care|service|support)\s+(?:executive|representative|agent|person|team)\b",
      rf"\b(?:connect|transfer|put|forward)\s+(?:me|this|us)\b[^.?!]{{0,30}}\b{_HUMAN_NOUN}\b",
      r"\bescalat(?:e|ed|ing|ion)\b",
      r"\b(?:call|ring|phone)\s+me\s+back\b", r"\b(?:someone|somebody|anyone)\s+(?:call|contact|phone)\b",
      r"\byour\s+(?:boss|superior)\b",
      rf"\b(?:get|give|call|fetch|want|need|demand|bring|ask for)\b[^.?!]{{0,15}}?{_NOT_POSSESSIVE}"
      rf"\b(?:manager|supervisor)\b(?!\s+(?:account|role|login|access|dashboard|portal|permission))",
      # Hinglish
      rf"{_NOT_POSSESSIVE}\b(?:manager|supervisor|agent|executive|representative|officer|insaan|insan|human)"
      rf"\s+se\s+(?:baat\s+{_KARNA}|connect\s+{_KARNA}|milna|milwa\w*|milao)\b",
      r"\bkisi\s+(?:insaan|insan|human|real|asli|agent|executive|representative|senior|officer)\b",
      rf"\bbaat\s+{_KARNA}\s+(?:thi|tha|hai|h)?\s*(?:manager|supervisor|agent|executive|insaan|human)\s+se\b",
      rf"\bkisi\s+se\s+baat\s+{_KARNA}",
      r"\b(?:real|asli)\s+(?:insaan|insan|person)\b",
      r"\bcustomer\s+care\s+(?:se|ko)\b",
      # Devanagari (no \b: it misbehaves around combining marks)
      r"इंसान\s+से\s+बात", r"(?:मैनेजर|सुपरवाइज़र|सुपरवाइजर|एजेंट)\s+से\s+(?:बात|कनेक्ट|मिल)",
      r"किसी\s+से\s+बात\s+(?:करनी|करना|कराओ|कराइए)")

_rule("legal", "escalation_trigger",
      r"\blegal\s+(?:action|notice|case|proceedings|steps|complaint)\b",
      r"\b(?:will|gonna|going to|'ll|shall|can|could|may|want to|ready to|about to|to)\s+sue\b",
      r"\bsue\s+(?:you|your|the|this|them|for)\b",
      r"\b(?:my|a|the|our|your|his|her)\s+(?:lawyer|attorney|advocate|solicitor)\b(?!-)",
      r"\bconsumer\s+(?:court|forum|commission|complaint|protection)\b", r"\bcourt\s+case\b",
      r"\bcourt\s+(?:jaunga|jaungi|jayenge|jaenge)\b", r"\blegal\s+(?:karwai|karyavahi)\b",
      r"\bpolice\s+(?:complaint|report|case|station)\b", r"\bcomplaint\s+to\s+(?:the\s+)?police\b",
      r"\bcyber\s?crime\b", r"\bvakeel\b", r"\bvakil\b",
      r"वकील", r"कंज्यूमर\s+कोर्ट", r"पुलिस")

_rule("fraud", "escalation_trigger",
      r"\bfraud(?:ulent|ster)?\b(?![- ]?(?:proof|protect\w*|prevent\w*|alert\w*))",
      r"\bscam(?:med|mers?)?\b(?![- ]?(?:proof|free|protect\w*))",
      r"\bcheat(?:ed|ing)\b", r"\b(?:crooks?|thieves|swindlers?)\b", r"\blog chor\b", r"\bdhokha\b", r"\bdhoka\b", r"\bthagi\b",
      r"धोखा", r"धोखाधड़ी", r"ठगी")


def check_input(text: str) -> SafetyResult:
    lowered = _normalize(text)
    for category, flag, pattern in _RULES:
        if pattern.search(lowered):
            return SafetyResult(safe=False, flag=flag, escalate=True, category=category)
    return SafetyResult(safe=True, flag=None, escalate=False)


# ---------------------------------------------------------------- output rules

# Response-side: language that suggests the LLM is promising something it cannot
# back. Only an issue when the answer is NOT grounded in retrieved context.
UNGROUNDED_PROMISE_PATTERNS = [
    r"\bguarantee(?:d|s)?\b", r"\bpromise(?:d|s)?\b",
    r"\b100\s?%\s?(?:refund|sure|certain|milega|pakka|guarantee\w*)",
    r"\b(?:refund|paisa|paise)\s+100\s?%",
    r"\bpakka\b", r"\bpukka\b",
]

# Figures an answer must not invent. Only units that make a POLICY claim are
# checked (days/weeks/months, money, percentages); "10 minutes while I check" or
# "9 am to 6 pm" is conversation, not policy, and must not trip the check.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "sixty": 60, "ninety": 90,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "panch": 5, "chhe": 6, "saat": 7,
    "aath": 8, "nau": 9, "das": 10, "pandrah": 15, "bees": 20, "tees": 30,
}
_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUM = r"(?:" + _DIGITS + "|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
_HOURS_PER_UNIT = {
    "hour": 1, "hours": 1, "hr": 1, "hrs": 1, "ghanta": 1, "ghante": 1,
    "day": 24, "days": 24, "din": 24,
    "week": 168, "weeks": 168, "hafta": 168, "hafte": 168,
    "month": 720, "months": 720, "mahina": 720, "mahine": 720,
    "year": 8760, "years": 8760,
}
_UNITS = "|".join(sorted(_HOURS_PER_UNIT, key=len, reverse=True))
_DURATION = re.compile(
    rf"\b({_NUM})(?:\s*(?:-|–|to|se|or)\s*({_NUM}))?\s*-?\s*(?:business |working |calendar )?({_UNITS})\b")
# "a week" / "an hour": the corpus may say it that way, so it counts as 1 there
# (never in the response: "a day or two" must not be read as a claim of one day).
_ARTICLE_DURATION = re.compile(rf"\b(?:a|an)\s+(?:business |working |calendar )?({_UNITS})\b")
_CURRENCY_PREFIX = r"(?:₹|\$|(?<![a-z])(?:rs\.?|inr|usd|rupees?))"
_CURRENCY_SUFFIX = r"(?:₹|\$|(?:rs\.?|inr|usd|rupees?|rupaye|rupaiye|rupiya|dollars?)(?![a-z]))"
_MONEY = re.compile(rf"{_CURRENCY_PREFIX}\s*({_DIGITS})|({_DIGITS})\s*{_CURRENCY_SUFFIX}")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent|pratishat)")


def _number(token: str) -> float:
    return float(_NUMBER_WORDS[token]) if token in _NUMBER_WORDS else float(token.replace(",", ""))


def _specifics(text: str, *, corpus: bool) -> set[tuple[str, float]]:
    """(kind, value) for every durable claim in `text`. Durations are normalised to
    hours so "1 week", "7 days" and "168 hours" are the same claim."""
    lowered = _normalize(text)
    found: set[tuple[str, float]] = set()
    for first, second, unit in _DURATION.findall(lowered):
        for token in (first, second):
            if token:
                found.add(("duration_hours", _number(token) * _HOURS_PER_UNIT[unit]))
    if corpus:
        for unit in _ARTICLE_DURATION.findall(lowered):
            found.add(("duration_hours", float(_HOURS_PER_UNIT[unit])))
    for prefixed, suffixed in _MONEY.findall(lowered):
        found.add(("money", _number(prefixed or suffixed)))
    for value in _PERCENT.findall(lowered):
        found.add(("percent", float(value)))
    return found


def unsupported_specifics(response: str, allowed_texts: Iterable[str]) -> list[tuple[str, float]]:
    """Figures stated in `response` that appear nowhere in `allowed_texts`.

    A money or percentage claim is also supported by the same number written WITHOUT
    its unit anywhere in the allowed text ("I paid 1499" supports "Rs 1,499"): people
    state amounts bare, and calling the agent's echo of them an invention would
    escalate honest answers. Durations are not relaxed this way, because a bare
    "30" says nothing about days versus hours."""
    texts = list(allowed_texts)
    allowed: set[tuple[str, float]] = set()
    bare: set[float] = set()
    for text in texts:
        allowed |= _specifics(text, corpus=True)
        bare |= {float(token.replace(",", "")) for token in re.findall(_DIGITS, _normalize(text))}
    missing = _specifics(response, corpus=False) - allowed
    return sorted(item for item in missing if not (item[0] in ("money", "percent") and item[1] in bare))


def _format(kind: str, value: float) -> str:
    return f"{kind}:{value:g}"


def check_output(text: str, grounded_in_kb: bool, context: Sequence[str] | None = None,
                 conversation: Sequence[str] = ()) -> SafetyResult:
    """Screen a generated answer.

    context       retrieved KB passages the answer was allowed to use. Passing it
                  (even as an empty list) turns on the figures check; callers that
                  omit it get only the promise check, as before.
    conversation  what the customer and the agent already said this session. A
                  figure the customer stated ("I paid Rs 1500") or the agent
                  already gave is not an invention.
    """
    lowered = _normalize(text)
    if not grounded_in_kb:
        for pattern in UNGROUNDED_PROMISE_PATTERNS:
            if re.search(pattern, lowered):
                return SafetyResult(safe=False, flag="ungrounded_promise", escalate=False, category="promise")

    if context is not None:
        missing = unsupported_specifics(text, [*context, *conversation])
        if missing:
            return SafetyResult(safe=False, flag="ungrounded_specifics", escalate=False, category="specifics",
                                detail=tuple(_format(kind, value) for kind, value in missing))
    return SafetyResult(safe=True, flag=None, escalate=False)
