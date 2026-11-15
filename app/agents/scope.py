"""
Scoped figures: "<figure> to <listed things> and <other figure> to all other ...".

    "Standard delivery takes 4-7 business days to metro cities (Delhi, Mumbai, Bengaluru, ...) and
     7-10 business days to all other locations."

The live evaluation (h-53, "Will my order reach Pune in 5 days?") showed the failure this module exists for:
retrieval was perfect, every number in the answer was in the context, and the answer was still wrong -- 4-7 days
was applied to a city that is not in the list. The numeric guard cannot see that: it asks "is this figure in the
context?", not "does it apply to THIS customer?". Prompt wording alone did not fix it either, because the decision
is a set-membership test that a small model performs implicitly, in one pass, with the customer's "5 days"
pulling it toward the nearer range.

So the membership test is done in code, twice, with no extra LLM call:
  * BEFORE generation, `scope_notes` states in plain words which figure applies to this conversation;
  * AFTER generation, `check_scope` flags an answer that gives the listed-scope figure (and not the other one)
    when the customer has named nothing on the list.

It is generic: the list may be cities, products, payment methods -- anything a tenant's document scopes with a
"(a, b, c) ... all other ..." sentence. It is deliberately narrow: only that tiered shape, so a parenthesised
list of examples or options never triggers it. Known limits (documented, not hidden):
  * a listed entity the customer names in a way neither the alias table, the typo tolerance nor the word match
    recognises is treated as "not named". The answer is then regenerated once with the list in front of the
    model and, if it still stands, handed to a human -- degraded, but never a confident wrong figure;
  * the reverse word order ("all other X take 7-10 days, while Y (a, b) take 4-7") is not parsed;
  * a rule split across two retrieved chunks is not seen.
"""

import difflib
import re
from dataclasses import dataclass

from app.agents.guardrails import SafetyResult, _normalize, _specifics

# One figure expression: a duration (incl. ranges), an amount, or a percentage.
_FIGURE = re.compile(
    r"(?:\d+(?:\.\d+)?(?:\s*(?:-|–|to)\s*\d+(?:\.\d+)?)?[\s-]*(?:business\s+|working\s+)?"
    r"(?:days?|hours?|hrs?|weeks?|months?)"
    r"|(?:rs\.?|inr|₹)\s*\d[\d,]*(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*%)",
    re.IGNORECASE)
_PAREN_LIST = re.compile(r"\(([^()]{3,300})\)")
_OTHER_MARKER = re.compile(
    r"\b(?:all|every|any)\s+other\b|\bthe\s+rest\b|\bremaining\b|\bother\s+"
    r"(?:locations|places|cities|areas|items|products|orders|cases|methods)\b", re.IGNORECASE)
_LIST_SPLIT = re.compile(r",|\band\b|\bor\b|/|&", re.IGNORECASE)
_LIST_PREFIX = re.compile(r"^\s*(?:e\.?g\.?|such as|like|incl\.?|including)\s*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")

# Indian-market name variants. Customers say Bangalore; documents say Bengaluru. Each group is one entity.
_ALIAS_GROUPS = (
    ("bengaluru", "bangalore", "bengalooru"), ("mumbai", "bombay"), ("kolkata", "calcutta"),
    ("chennai", "madras"), ("delhi", "new delhi", "dilli"), ("gurugram", "gurgaon"), ("pune", "poona"),
    ("kochi", "cochin"), ("thiruvananthapuram", "trivandrum"), ("varanasi", "benares", "banaras"),
    ("vadodara", "baroda"), ("mysuru", "mysore"), ("hyderabad", "secunderabad"),
)
_ALIASES: dict[str, set[str]] = {}
for _group in _ALIAS_GROUPS:
    for _name in _group:
        _ALIASES.setdefault(_name, set()).update(_group)

FUZZY_RATIO = 0.85   # "Mumbay" ~ "mumbai"; high enough that "pune" never matches "puna"-like unrelated words


@dataclass(frozen=True)
class ScopedRule:
    entities: tuple[str, ...]
    scoped_text: str
    other_text: str
    scoped_figures: frozenset
    other_figures: frozenset


def _sentence_around(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for m in _SENTENCE_END.finditer(text, 0, start):
        left = m.end()
    m = _SENTENCE_END.search(text, end)
    return left, (m.start() if m else len(text))


def _entities(raw: str) -> tuple[str, ...]:
    items = []
    for part in _LIST_SPLIT.split(_LIST_PREFIX.sub("", raw)):
        part = _normalize(part).strip(" .;:-")
        if 2 <= len(part) <= 40 and re.search(r"[a-z]", part):
            items.append(part)
    return tuple(items) if len(items) >= 2 else ()


def extract_scoped_rules(context: list[str]) -> list[ScopedRule]:
    """Every "<figure> ... (list) ... <other figure> ... all other ..." sentence in the retrieved passages."""
    rules: list[ScopedRule] = []
    for passage in context:
        for paren in _PAREN_LIST.finditer(passage):
            entities = _entities(paren.group(1))
            if not entities:
                continue
            s_start, s_end = _sentence_around(passage, paren.start(), paren.end())
            before = list(_FIGURE.finditer(passage, s_start, paren.start()))
            after = passage[paren.end():s_end]
            marker = _OTHER_MARKER.search(after)
            if not marker:
                continue
            figures_after = list(_FIGURE.finditer(after))
            # The scoped figure usually precedes the list ("4-7 days to metros (a, b)"), but may follow it
            # ("Electronics (a, b) have a 10-day window, while all other items ... 30 days").
            if before:
                scoped_m = before[-1]
                pool = figures_after
            elif figures_after:
                scoped_m, pool = figures_after[0], figures_after[1:]
            else:
                continue
            nearest = [m for m in pool if m.end() <= marker.start()]
            other_match = nearest[-1] if nearest else next((m for m in pool if m.start() >= marker.end()), None)
            if other_match is None:
                continue
            scoped, other = scoped_m.group(0).strip(), other_match.group(0).strip()
            scoped_f, other_f = frozenset(_specifics(scoped, corpus=False)), frozenset(_specifics(other, corpus=False))
            if scoped_f and other_f and scoped_f != other_f:
                rules.append(ScopedRule(entities, scoped, other, scoped_f, other_f))
    return rules


def _mentions(entity: str, text: str, words: list[str]) -> bool:
    """Whole-word match on the entity or an alias, with tolerance for a typo ("Mumbay")."""
    for variant in _ALIASES.get(entity, {entity}) | {entity}:
        if re.search(rf"(?<![a-z]){re.escape(variant)}(?![a-z])", text):
            return True
        if " " not in variant and len(variant) >= 5:
            for word in words:
                if (len(word) >= 5 and abs(len(word) - len(variant)) <= 2
                        and difflib.SequenceMatcher(None, word, variant).ratio() >= FUZZY_RATIO):
                    return True
    return False


def _named(rule: ScopedRule, conversation: list[str]) -> list[str]:
    text = _normalize(" ".join(conversation))
    words = re.findall(r"[a-z]+", text)
    return [e for e in rule.entities if _mentions(e, text, words)]


def scope_notes(context: list[str], conversation: list[str]) -> list[str]:
    """Plain-words statements, one per rule, of which figure applies to THIS conversation. Appended to the
    prompt's context so the model is told the answer to the membership test instead of having to do it."""
    notes = []
    for rule in extract_scoped_rules(context):
        listed = ", ".join(rule.entities)
        named = _named(rule, conversation)
        if named:
            notes.append(f'"{rule.scoped_text}" applies only to: {listed}. The customer named {", ".join(named)}, '
                         f"so it applies to them.")
        else:
            notes.append(
                f'"{rule.scoped_text}" applies ONLY to: {listed}. For everything else the figure is '
                f'"{rule.other_text}". The customer has not named any of those; unless what they mention is one of '
                f'them under another name or spelling, use "{rule.other_text}". If the customer has not said '
                f"which applies to them, give both cases or ask.")
    return notes


def scope_violations(answer: str, context: list[str], conversation: list[str]) -> list[ScopedRule]:
    """Rules whose listed-scope figure the answer gives (without the other figure) although the CUSTOMER named
    nothing on the list. The answer's own words never count as evidence: a model that invents "Mumbai" must not
    excuse itself."""
    stated = _specifics(answer, corpus=False)
    violated = []
    for rule in extract_scoped_rules(context):
        if rule.scoped_figures <= stated and not rule.other_figures <= stated and not _named(rule, conversation):
            violated.append(rule)
    return violated


def check_scope(answer: str, context: list[str], conversation: list[str]) -> SafetyResult | None:
    """A SafetyResult for the output pipeline, or None when the answer is fine. Carries no text."""
    if scope_violations(answer, context, conversation):
        return SafetyResult(safe=False, flag="scope_violation", escalate=False, category="scope")
    return None
