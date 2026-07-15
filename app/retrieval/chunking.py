"""
Chunking for document ingestion.

Why ~400-800 tokens (~300-600 words) with ~15% overlap rather than
one embedding per document or per sentence:
Too large (whole document) dilutes retrieval precision -- a query
about refunds pulls back a 10-page doc where refunds are one
paragraph, and that whole doc gets stuffed into the prompt. Too small
(single sentence) loses surrounding context needed to answer
correctly. Overlap exists so a fact split across a chunk boundary
(e.g. "Refunds take 5-7 days." | "This applies to all payment
methods.") doesn't lose either half depending on where the cut falls.
Word-count is used here as a cheap proxy for token count (roughly
0.75 words per token in English/Hinglish text) -- swap for a real
tokenizer if chunk-size precision becomes a measured problem.
"""


def chunk_text(text: str, target_words: int = 450, overlap_words: int = 70) -> list[str]:
    words = text.split()
    if len(words) <= target_words:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = start + target_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words  # step back by the overlap for the next chunk

    return chunks
