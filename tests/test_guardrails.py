from app.agents.guardrails import check_input, check_output


class TestInputGuardrails:
    def test_clean_message_passes(self):
        result = check_input("Where is my order?")
        assert result.safe is True
        assert result.escalate is False
        assert result.flag is None

    def test_legal_threat_triggers_escalation(self):
        result = check_input("I will sue you if this isn't fixed")
        assert result.safe is False
        assert result.escalate is True
        assert result.flag == "escalation_trigger"

    def test_manager_request_triggers_escalation(self):
        result = check_input("Let me speak to a manager")
        assert result.escalate is True

    def test_self_harm_language_triggers_escalation(self):
        result = check_input("I feel like I want to kill myself over this")
        assert result.escalate is True

    def test_case_insensitive_matching(self):
        result = check_input("I NEED A LAWYER for this fraud")
        assert result.escalate is True


class TestOutputGuardrails:
    def test_grounded_response_with_promise_language_passes(self):
        # If it IS grounded in KB, promise-like language is acceptable
        result = check_output("We guarantee refunds within 7 days per policy", grounded_in_kb=True)
        assert result.safe is True

    def test_ungrounded_guarantee_is_flagged(self):
        result = check_output("I guarantee you'll get double the refund", grounded_in_kb=False)
        assert result.safe is False
        assert result.flag == "ungrounded_promise"

    def test_ungrounded_neutral_response_passes(self):
        result = check_output("Let me check that for you and get back to you.", grounded_in_kb=False)
        assert result.safe is True


# =============================================================================
# Guardrails v2
# =============================================================================
import pytest

from app.agents.guardrails import unsupported_specifics
from app.evaluation.guardrail_eval import evaluate_input, evaluate_output, load_cases

CASES = load_cases()


def test_the_labelled_input_set_has_no_regressions():
    """A REGRESSION suite: the cases are self-written (see the file's _about), so a
    perfect score here is not an accuracy claim -- it says no known case got worse."""
    report = evaluate_input(CASES["input"])
    assert report.errors == [], report.errors
    assert report.precision == 1.0 and report.recall == 1.0


def test_the_labelled_output_set_has_no_regressions():
    report = evaluate_output(CASES["output"])
    assert report.errors == [], report.errors


def test_the_set_covers_the_languages_the_product_serves():
    kinds = {c["kind"] for c in CASES["input"]}
    assert {"human_request_hinglish", "legal_hinglish", "human_request_devanagari", "self_harm_hinglish"} <= kinds
    assert sum(1 for c in CASES["input"] if c["expect"] == "pass") >= 30      # hard negatives are most of the point


@pytest.mark.parametrize("text", [
    "my manager told me to order these",         # possessive: the customer's own boss
    "This is Sue, I need help with my order",    # a name, not a threat
    "Do you sell fraud-proof wallets?",          # a product
    "Can I return the lawyer-grade briefcase?",  # a product
    "I need a manager account on the dashboard", # a role, not a request
    "I want to die laughing at this meme",       # an idiom
    "Sue Ellen placed the order",
])
def test_innocent_text_that_used_to_trip_a_keyword_rule_passes(text):
    assert check_input(text).safe, text


def test_self_harm_has_its_own_flag_and_wins_over_other_rules():
    result = check_input("this is fraud and I want to kill myself")
    assert result.flag == "self_harm" and result.category == "self_harm" and result.escalate
    assert check_input("I will sue you").flag == "escalation_trigger"      # existing flag unchanged


def test_input_matching_survives_unicode_width_case_and_whitespace_tricks():
    assert not check_input("ＳＰＥＡＫ   TO   A   HUMAN   AGENT").safe       # full-width, caps, spacing


def test_the_matched_text_is_never_part_of_the_result():
    result = check_input("kisi insaan se baat karni hai")
    assert "insaan" not in repr(result)      # logged as a category, so message content never reaches logs


class TestSpecificsCheck:
    def test_legacy_callers_without_context_get_only_the_promise_check(self):
        assert check_output("Refunds take 5 days.", grounded_in_kb=False).safe

    def test_an_invented_figure_is_flagged_with_what_was_unsupported(self):
        result = check_output("You will get Rs 5000 in 2 days.", grounded_in_kb=False, context=[])
        assert result.flag == "ungrounded_specifics" and set(result.detail) == {"money:5000", "duration_hours:48"}

    @pytest.mark.parametrize("response,context", [
        ("Refunds take 7 days.", ["Returns are accepted within 1 week."]),            # week = 7 days
        ("It takes 24 hours.", ["Replacements ship within 1 day."]),
        ("Delivery takes 3 days.", ["Delivery takes three days."]),                   # number word
        ("Aapka refund 5 se 7 din me aayega.", ["Refunds take 5-7 business days."]),  # Hinglish phrasing
        ("Return within 30 days.", ["We have a 30-day return policy."]),              # hyphenated compound
        ("Return within 30 days.", ["Valid for a month."]),                           # "a month" in the KB = 1 month
        ("It costs $20.", ["Priced at 20 dollars."]),
        ("Rs 1,499 will be refunded.", ["Refund equals the price paid."]),            # customer's bare amount, below
    ])
    def test_equivalent_phrasings_are_not_false_alarms(self, response, context):
        said = ["I paid 1499 for it"]
        assert unsupported_specifics(response, [*context, *said]) == [], (response, context)

    def test_conversation_filler_and_clock_times_are_not_policy_claims(self):
        assert unsupported_specifics("Give me 10 minutes. We are open 9 am to 6 pm, 24x7 online.", []) == []

    def test_a_range_is_checked_endpoint_by_endpoint(self):
        assert unsupported_specifics("It takes 4 days.", ["Delivery takes 3-5 days."]) != []   # 4 was never stated

    def test_a_response_phrase_like_a_day_or_two_is_not_read_as_a_claim(self):
        assert unsupported_specifics("You will have it in a day or two.", []) == []

    def test_a_wrong_figure_next_to_the_right_unit_is_caught(self):
        assert unsupported_specifics("Free shipping above Rs 500.", ["Free shipping above Rs.5,000"]) != []

    def test_a_percentage_is_checked(self):
        assert unsupported_specifics("You get 50% cashback.", ["Get 5% cashback"]) != []

    def test_hinglish_promise_words_are_caught_when_ungrounded(self):
        assert check_output("Refund pakka milega", grounded_in_kb=False).flag == "ungrounded_promise"
        assert check_output("Refund pakka milega", grounded_in_kb=True).safe      # grounded: allowed, as before
