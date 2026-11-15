
import pytest

from app.evaluation.language import detect_language, language_matches
from app.evaluation.rag_eval import HARD_DATASET_PATH, load_dataset

REPORT_ENGLISH_Q_HINGLISH_A = [   # (report) English questions that got Hinglish answers
    ("How long does a refund take for a Cash on Delivery order?",
     "COD orders ke liye refund process hone mein 7-10 business days lagte hain, once humein returned item mil jaye."),
    ("I paid online, in how many days will my refund reach my card?",
     "Prepaid orders ke liye refund aapke original payment method par 5-7 business days mein process ho jata hai."),
    ("They delivered the wrong item. Do I pay for the return?",
     "Arre nahi, bilkul nahi! Wrong item ke case mein aapko koi extra charge nahi dena padega."),
    ("What is the return window for a laptop?",
     "Laptop ke liye 10-day return window hoti hai, but make sure ki woh sealed box mein ho."),
]
REPORT_ENGLISH_Q_ENGLISH_A = [
    ("Will my shipping charge be refunded?",
     "Shipping charges are generally non-refundable, but if you received a defective, damaged, or wrong item, we will refund it."),
    ("Can I return sneakers after 10 days?",
     "Hey, I'm sorry but footwear can only be returned within 7 days of delivery, provided they are unworn."),
]


@pytest.mark.parametrize("question,answer", REPORT_ENGLISH_Q_HINGLISH_A)
def test_a_hinglish_answer_to_an_english_question_is_a_mismatch(question, answer):
    assert detect_language(question) == "english" and detect_language(answer) == "hinglish"
    assert not language_matches(question, answer)


@pytest.mark.parametrize("question,answer", REPORT_ENGLISH_Q_ENGLISH_A)
def test_an_english_answer_to_an_english_question_matches(question, answer):
    assert language_matches(question, answer)


@pytest.mark.parametrize("text,expected", [
    ("Order cancel kaise karun?", "hinglish"),                       # short: one marker is enough
    ("Kya aap gift wrapping karte ho?", "hinglish"),
    ("Payment cut ho gaya lekin order fail dikha raha hai", "hinglish"),
    ("aur kitne din lagenge?", "hinglish"),
    ("मुझे रिफंड चाहिए", "devanagari"),
    ("hw do i trak my ordr", "english"),                              # typos are not a language switch
    ("I forgot my password, how do I reset it?", "english"),
    ("Can I pay with Bitcoin?", "english"),
    ("Please do not pay again to the main account, ok?", "english"),  # 'do', 'to', 'main', 'ok' are English
    ("", "english"),
])
def test_register_detection_on_representative_messages(text, expected):
    assert detect_language(text) == expected


def test_detection_agrees_with_the_labelled_kinds_of_the_hard_dataset():
    """The dataset labels hinglish questions as such; no English-kind question may be read as Hinglish
    (a detector that over-fires would invent language mismatches in the report)."""
    cases = load_dataset(HARD_DATASET_PATH)
    hinglish = [c for c in cases if c["kind"] == "hinglish"]
    assert hinglish and all(detect_language(c["question"]) == "hinglish" for c in hinglish), \
        [c["question"] for c in hinglish if detect_language(c["question"]) != "hinglish"]
    english = [c for c in cases if c["kind"] in ("direct", "typo", "long-doc", "category")]
    wrong = [c["question"] for c in english if detect_language(c["question"]) != "english"
             and not any(w in c["question"].lower() for w in ("kab", "kitne", "kya", "mera", "ho "))]
    assert wrong == [], wrong
