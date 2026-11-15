"""
A CONTRACT test for the system prompt: it pins the rules the first live evaluation showed were needed, so a
later prompt edit cannot silently drop one. It does NOT prove the model obeys them -- only a real run does
(`python -m app.evaluation.rag_eval --corpus realistic`, which now scores language match and unsupported
refusals). Each rule below maps to a finding in that evaluation.
"""

import pytest

from app.prompts.templates import SYSTEM_PROMPT, build_response_prompt

RULES = {
    "language follows the LATEST message": "language of the customer's latest message",
    "English gets plain English (h-01/02/03/05/08 got Hinglish)": "plain english",
    "no unprompted Hinglish fillers for English speakers": '"arre", "haanji"',
    "Devanagari gets Devanagari": "devanagari",
    "refusals need context support (h-47 Bitcoin, h-49 gift wrapping)": "unless the context explicitly says so",
    "not mentioned != not offered": "not the same as not being offered",
    "no assumed categories (h-14 Jaipur called a metro)": 'a city is not a "metro"',
    "unlisted city/product -> general rule": "use the general rule",
    "original rules kept: no invented facts": "never invent policy details",
    "original rules kept: no promises": "never make promises",
}


@pytest.mark.parametrize("why,needle", RULES.items())
def test_the_prompt_keeps_each_rule_the_evaluation_showed_was_needed(why, needle):
    assert needle in " ".join(SYSTEM_PROMPT.lower().split()), why          # whitespace-insensitive: wrapping is free to change


def test_the_old_instruction_that_biased_replies_toward_hinglish_is_gone():
    assert "don't force pure english" not in SYSTEM_PROMPT.lower()


def test_the_context_is_still_appended_and_the_empty_case_is_explicit():
    system = build_response_prompt(["refund in 7 days"], [], "q")[0]["content"]
    assert system.startswith(SYSTEM_PROMPT) and "refund in 7 days" in system
    assert "No relevant context found." in build_response_prompt([], [], "q")[0]["content"]
