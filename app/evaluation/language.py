"""
A small, dependency-free language-register detector for the RAG evaluation: English, romanised Hindi
("Hinglish") or Devanagari. It exists because the first live evaluation showed English questions
getting Hinglish answers and the LLM judge never noticed -- so this check is deterministic and
separate from the judge.

It is a heuristic over a list of common romanised-Hindi function words, deliberately restricted to
words that are not ordinary English ("do", "to", "me", "main", "par" are excluded). It is good for
measuring a drift RATE across dozens of answers, not for classifying one arbitrary sentence.
"""

import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[a-z']+")

HINGLISH_MARKERS = frozenset("""
hai hain ho hoga hogi honge hoon hota hoti hue hua tha thi the_ nahi nahin nhi toh kya kab kaise kaisa kyun kyu
kitne kitna kitni kahan kaun mera meri mere mujhe mujhko hum humein humara hamare hamara aap aapka aapki aapke
aapko aapne apna apni apne tum tumhara iska iski uska uski yeh woh wo ye
kar karo karna karne karte karta karein kijiye kijiyega karun karunga karwa karwaun karen
mein mai se ko ka ki ke liye tak bhi hi aur lekin par_ agar jab tab abhi bas sirf bilkul
jayega jayegi jaye jaata jaati sakte sakta sakti sakti chahiye chahta chahte raha rahi rahe gaya gayi gaye
diya diye liya liye_ mila mili milega milegi milta lagenge lagega lagta lagte
din ghante ghanta baad pehle wapas dobara phir ab yahan wahan koi kuch sab bahut thoda zyada
paise paisa rupaye order_ bhejna bhejo dena dete lena lete batao bataiye bataye dekho dekhiye
arre haan haanji namaste dhanyavaad shukriya
""".replace("_", "x_").split())  # trailing-underscore entries are deliberately inert (ambiguous with English)

SHORT_TEXT_WORDS = 4  # one marker is enough in a message this short ("Order cancel kaise karun?")


def detect_language(text: str) -> str:
    """'devanagari', 'hinglish' or 'english'."""
    if _DEVANAGARI.search(text):
        return "devanagari"
    words = _WORD.findall(text.lower().replace("’", "'"))
    hits = sum(1 for w in words if w in HINGLISH_MARKERS)
    if hits >= 2 or (hits >= 1 and len(words) <= SHORT_TEXT_WORDS):
        return "hinglish"
    return "english"


def language_matches(question: str, answer: str) -> bool:
    """The answer is in the register of the question (the rule in SYSTEM_PROMPT)."""
    return detect_language(question) == detect_language(answer)
