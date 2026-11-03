"""
RAG retrieval layer -- pgvector-backed, tenant-scoped.

Why pgvector over a dedicated vector DB:
See app/db/models.py docstring for the full reasoning. Short version:
one fewer moving part, one transactional boundary shared with the
rest of the tenant data, and a genuinely defensible "didn't reach for
extra infra before it was needed" position.

Why every query here takes business_id and filters on it directly in
SQL (not "fetch top-k globally, then filter in Python"):
Filtering after the fact means the ANN index still has to consider
every tenant's vectors to find the top-k, which is slower AND means a
filtering bug anywhere downstream leaks tenant data. Filtering inside
the SQL WHERE clause means the database itself enforces the tenant
boundary -- there's no code path that can accidentally skip it.

Why Google Gemini embeddings (gemini-embedding-001) instead of OpenAI:
Free via an AI Studio API key (no card required), which matters for a
student project that shouldn't need billing setup to run. Also
natively multilingual across 100+ languages including Hindi, which
fits Hinglish content better than defaulting to an English-first
model. output_dimensionality is pinned to 1536 (config.embedding_dim)
to match the pgvector column -- this model defaults to 3072 dims but
supports truncating via Matryoshka Representation Learning without a
meaningful quality loss. Swappable via `embed_text` alone if that
ever changes.

Why task_type differs between indexing and querying (RETRIEVAL_DOCUMENT
vs RETRIEVAL_QUERY):
Gemini's embedding model is trained to place a query and its matching
document close in vector space *asymmetrically* -- a question and its
answer aren't phrased the same way, so embedding both with the same
generic setting gives measurably worse retrieval than telling the
model which side of the search each text represents.
"""

from dataclasses import dataclass
import uuid

import structlog
from google import genai
from google.genai import types
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.upstream import UpstreamEmptyResponse, call_with_retry
from app.db.models import KnowledgeChunk

logger = structlog.get_logger()
_genai_client = genai.Client(api_key=settings.google_api_key)

# Texts per embedding request when ingesting a document.
EMBED_BATCH_SIZE = 32


@dataclass
class RetrievedDoc:
    id: str
    text: str
    score: float
    metadata: dict


async def _embed(contents: str | list[str], task_type: str) -> list:
    """One embedding request under the shared upstream policy (timeout, bounded
    retry of transient errors only). Raises Upstream* errors with service="embedding"."""
    async def call():
        return await _genai_client.aio.models.embed_content(
            model=settings.embedding_model,
            contents=contents,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embedding_dim,
            ),
        )

    result = await call_with_retry("embedding", call)
    if not result.embeddings:
        raise UpstreamEmptyResponse("embedding", "empty_response")
    return result.embeddings


async def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    return (await _embed(text, task_type))[0].values


async def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed many texts, EMBED_BATCH_SIZE per request, preserving order.

    If the API ever returns a different number of embeddings than inputs, the
    batch is redone one text at a time rather than risking misaligned vectors
    (a chunk silently stored with another chunk's embedding)."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        embeddings = await _embed(batch, task_type)
        if len(embeddings) != len(batch):
            logger.warning("embed_batch_count_mismatch", expected=len(batch), received=len(embeddings))
            embeddings = [(await _embed(text, task_type))[0] for text in batch]
        vectors.extend(e.values for e in embeddings)
    return vectors


async def upsert_chunk(db: AsyncSession, document_id: uuid.UUID, business_id: uuid.UUID,
                        text: str, metadata: dict) -> KnowledgeChunk:
    vector = await embed_text(text, task_type="RETRIEVAL_DOCUMENT")
    chunk = KnowledgeChunk(
        document_id=document_id,
        business_id=business_id,
        content=text,
        embedding=vector,
        chunk_metadata=metadata,
    )
    db.add(chunk)
    await db.commit()
    await db.refresh(chunk)
    return chunk


async def retrieve(db: AsyncSession, business_id: uuid.UUID, query: str,
                    top_k: int = 3, score_threshold: float = 0.25) -> list[RetrievedDoc]:
    """
    Returns the top_k most relevant KB chunks for a query, scoped to one
    business_id only.

    score_threshold guards against forcing irrelevant context into the
    prompt when the tenant's KB has nothing relevant -- better to let
    the LLM say "let me check on that" than hallucinate off a weak
    match. Cosine distance in pgvector (`<=>`) is 0 for identical
    vectors and increases with dissimilarity, so we convert to a
    similarity score (1 - distance) to keep the same threshold
    semantics as before.
    """
    distance = KnowledgeChunk.embedding.cosine_distance(await embed_text(query, task_type="RETRIEVAL_QUERY"))
    stmt = (
        select(KnowledgeChunk, distance.label("distance"))
        .where(KnowledgeChunk.business_id == business_id)
        .order_by(distance)
        .limit(top_k)
    )
    result = await db.execute(stmt)
    rows = result.all()

    docs = []
    for chunk, dist in rows:
        similarity = 1 - dist
        if similarity >= score_threshold:
            docs.append(RetrievedDoc(
                id=str(chunk.id),
                text=chunk.content,
                score=similarity,
                metadata=chunk.chunk_metadata,
            ))
    return docs
