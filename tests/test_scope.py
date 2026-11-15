"""
Scoped-figure rules: "<figure> to <listed things> and <other figure> to all other ...".

Regression for the live-eval failure h-53 ("Will my order reach Pune in 5 days?"): retrieval was right, the
context said 4-7 days applies to six listed metros and 7-10 days to every other location, and the model still
answered 4-7. The numeric guard could not catch it because 4-7 days IS in the context -- the error was applying
a real figure to the wrong case. These tests are generic on purpose: the same logic must hold for products,
payment methods or any other listed scope, not just cities.
"""

import pytest

from app.agents.scope import check_scope, extract_scoped_rules, scope_notes, scope_violations

SHIPPING = ("Standard delivery takes 4-7 business days to metro cities (Delhi, Mumbai, Bengaluru, Chennai, "
            "Kolkata, Hyderabad) and 7-10 business days to all other locations.")
RETURNS = ("Electronics (laptops, phones, tablets) have a 10-day return window, while all other items can be "
           "returned within 30 days.")
CASHBACK = "Get 5% cashback on orders paid by UPI (GPay, PhonePe, Paytm) and 2% on all other payment methods."


# ------------------------------------------------------------------ extraction

def test_the_shipping_rule_is_extracted_with_both_figures_and_the_full_list():
    (rule,) = extract_scoped_rules([SHIPPING])
    assert rule.entities == ("delhi", "mumbai", "bengaluru", "chennai", "kolkata", "hyderabad")
    assert rule.scoped_text == "4-7 business days" and rule.other_text == "7-10 business days"
    assert ("duration_hours", 96.0) in rule.scoped_figures and ("duration_hours", 240.0) in rule.other_figures


@pytest.mark.parametrize("text,scoped,other", [
    (RETURNS, "10-day", "30 days"),
    (CASHBACK, "5%", "2%"),
])
def test_the_same_pattern_is_found_for_products_and_payment_methods(text, scoped, other):
    (rule,) = extract_scoped_rules([text])
    assert rule.scoped_text == scoped and rule.other_text == other


@pytest.mark.parametrize("text", [
    "Express delivery is available in 20 cities for Rs 99.",                              # no list
    "Delivery to metro cities (Delhi, Mumbai) is fast.",                                  # list, but no figures/tiers
    "Refunds go to the original method (UPI, cards, net banking) in 5-7 business days.",  # list of options, no 'all other' tier
    "NEFT (bank transfer) takes 3 days and all other modes take 5 days.",                 # one item is not a list
    "",
])
def test_text_without_a_tiered_scope_yields_no_rule(text):
    assert extract_scoped_rules([text]) == []


def test_rules_are_found_across_several_retrieved_passages():
    rules = extract_scoped_rules(["Unrelated passage.", SHIPPING, RETURNS])
    assert len(rules) == 2


# ------------------------------------------------------------------ pre-generation notes

def test_when_the_customer_names_no_listed_place_the_note_says_the_other_figure_applies():
    (note,) = scope_notes([SHIPPING], ["Will my order reach Pune in 5 days?"])
    assert '"4-7 business days" applies ONLY to' in note and "Delhi" in note.title() or "delhi" in note
    assert '"7-10 business days"' in note and "has not named" in note
    assert "another name or spelling" in note                       # alias escape hatch (Bangalore = Bengaluru)
    assert "give both cases or ask" in note


def test_when_the_customer_names_a_listed_place_the_note_says_the_scoped_figure_applies():
    (note,) = scope_notes([SHIPPING], ["Delivery to Mumbai kitne din mein hoga?"])
    assert "named mumbai" in note.lower() and "applies to them" in note


@pytest.mark.parametrize("message", ["I live in Bangalore", "send it to Bombay", "Mumbay ke liye kitne din?", "ship to CHENNAI"])
def test_aliases_typos_and_case_are_recognised_as_listed_places(message):
    (note,) = scope_notes([SHIPPING], [message])
    assert "applies to them" in note, message


@pytest.mark.parametrize("message", ["Will it reach Pune?", "Gurgaon delivery?", "Delhiwala store near me", "Chennaiexpress parcel"])
def test_unlisted_places_and_lookalike_words_are_not_matched(message):
    (note,) = scope_notes([SHIPPING], [message])
    assert "has not named" in note, message


def test_a_place_named_earlier_in_the_conversation_counts():
    (note,) = scope_notes([SHIPPING], ["I am in Hyderabad", "ok", "how long will it take?"])
    assert "applies to them" in note


def test_no_notes_when_the_context_has_no_tiered_rule():
    assert scope_notes(["Refunds take 5-7 business days."], ["hi"]) == []


# ------------------------------------------------------------------ post-generation check

@pytest.mark.parametrize("answer", [
    "Haan, Pune ke liye 4-7 business days lagte hain.",                          # Hinglish
    "Yes, your order to Pune will arrive in 4-7 business days, so 5 days is fine.",
    "Delivery usually takes 4-7 business days.",                                  # no place named anywhere
])
def test_a_scoped_figure_given_to_someone_outside_the_list_is_a_violation(answer):
    assert scope_violations(answer, [SHIPPING], ["Will my order reach Pune in 5 days?"]) != []
    result = check_scope(answer, [SHIPPING], ["Will my order reach Pune in 5 days?"])
    assert result.flag == "scope_violation" and result.category == "scope" and not result.escalate


@pytest.mark.parametrize("answer,conversation", [
    ("Pune is not one of the metro cities, so delivery takes 7-10 business days.", ["Will my order reach Pune in 5 days?"]),
    ("Mumbai is a metro, so it takes 4-7 business days.", ["Delivery to Mumbai?"]),
    ("Bangalore ke liye 4-7 business days lagte hain.", ["I'm in Bangalore, how long?"]),     # alias of Bengaluru
    ("Metro cities take 4-7 business days and all other locations take 7-10 business days. Where are you?",
     ["How long will delivery take?"]),                                                       # both cases stated
    ("Delivery to Gurgaon takes 7-10 business days.", ["Gurgaon?"]),
    ("Express delivery takes 1-2 business days.", ["Pune?"]),                                  # a different fact entirely
])
def test_correct_answers_are_not_flagged(answer, conversation):
    assert scope_violations(answer, [SHIPPING], conversation) == [], answer
    assert check_scope(answer, [SHIPPING], conversation) is None


def test_the_answer_naming_a_listed_place_does_not_vouch_for_itself():
    """Evidence must come from the CUSTOMER's side: a model that invents 'Mumbai' must not excuse itself."""
    answer = "For Mumbai it takes 4-7 business days."
    assert scope_violations(answer, [SHIPPING], ["Will my order reach Pune?"]) != []


def test_the_same_logic_protects_products_and_payment_methods():
    assert scope_violations("Sneakers have a 10-day return window.", [RETURNS], ["Can I return my sneakers?"]) != []
    assert scope_violations("Your laptop has a 10-day return window.", [RETURNS], ["Can I return my laptop?"]) == []
    assert scope_violations("You get 5% cashback.", [CASHBACK], ["I paid by card"]) != []
    assert scope_violations("You get 5% cashback.", [CASHBACK], ["I paid by PhonePe"]) == []


def test_the_check_is_inert_without_a_tiered_rule_or_without_context():
    assert check_scope("Refunds take 5-7 business days.", ["Refunds take 5-7 business days."], ["q"]) is None
    assert check_scope("It takes 4-7 days.", [], ["q"]) is None


def test_the_result_carries_no_answer_or_customer_text():
    result = check_scope("Pune takes 4-7 business days.", [SHIPPING], ["Will my order reach Pune?"])
    assert "pune" not in repr(result).lower()
