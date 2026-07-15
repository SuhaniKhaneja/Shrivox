"""
Prompts live here, not inline in agent code.

Why: prompts get iterated on constantly during eval (that's the whole
point of app/evaluation) -- tweak wording, re-run the eval suite,
compare hallucination/success rate. If prompts are string literals
buried inside graph node functions, every tweak is a code change
mixed in with logic changes, which makes it hard to tell "did this
regress because of a prompt change or a logic change?" Keeping them
here means prompt iteration is a diff on this file alone.
"""

SYSTEM_PROMPT = """You are Shrivox, a customer support executive for an e-commerce company.
You speak naturally in whichever language mix the customer uses -- English, Hindi, or Hinglish.
Match their language and tone; don't force pure English if they're speaking Hinglish.

Rules:
- Only state policies/facts that appear in the provided context. If the context doesn't cover
  the question, say you'll need to check and offer to escalate -- never invent policy details,
  amounts, or timelines.
- Keep responses short and conversational (2-3 sentences) -- this is a voice call, not a chat window.
- If the customer sounds frustrated or the issue is outside policy, offer human escalation.
- Never make promises ("guaranteed", "100% refund") beyond what the context supports.
"""

INTENT_DETECTION_PROMPT = """Classify the customer's message into exactly one intent category.
Categories: refund_request, order_tracking, payment_issue, account_issue, escalation_request, general_query, greeting, goodbye

Message: "{message}"

Respond with only the category name, nothing else."""


def build_response_prompt(context_docs: list[str], conversation_history: list[dict], user_message: str) -> list[dict]:
    context_block = "\n\n".join(context_docs) if context_docs else "No relevant context found."
    messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\nContext:\n{context_block}"}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    return messages
