"""
Knowledge-base document endpoints, scoped per business.

Phase 1 accepts raw text directly (JSON body) rather than file upload
+ PDF/DOCX parsing -- that's a real but separate concern (text
extraction quality varies wildly by format) that doesn't change
anything about the retrieval architecture. Swapping this endpoint to
accept multipart file uploads and route through a parser is additive,
not a redesign.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.config import settings
from app.db.models import Document, KnowledgeChunk
from app.retrieval.chunking import chunk_text
from app.retrieval.retriever import embed_texts
from app.api.deps import get_current_user, CurrentUser

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class DocumentUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)  # Document.filename is String(255)
    text: str

    @field_validator("text")
    @classmethod
    def _text_within_limits(cls, value: str) -> str:
        # Read at call time so the limit is configurable (MAX_DOCUMENT_CHARS).
        if len(value) > settings.max_document_chars:
            raise ValueError(f"text exceeds the {settings.max_document_chars} character limit")
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


@router.post("/upload", status_code=201)
async def upload_document(payload: DocumentUploadRequest, current_user: CurrentUser = Depends(get_current_user),
                           db: AsyncSession = Depends(get_db)):
    chunks = chunk_text(payload.text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document contains no usable text")

    # ALL external calls happen BEFORE anything is written. If embedding fails
    # (after bounded retries) the request fails with 503/502 and leaves NOTHING
    # behind -- no document stuck in 'processing', no orphaned chunks -- and no
    # database transaction is held open while waiting on Gemini.
    vectors = await embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")

    # The document and every chunk are written in ONE transaction: all or nothing.
    document = Document(
        id=uuid.uuid4(), business_id=current_user.business_id,
        filename=payload.filename, file_type="text", status="ready",
    )
    db.add(document)
    db.add_all([
        KnowledgeChunk(
            document_id=document.id, business_id=current_user.business_id,
            content=chunk, embedding=vector, chunk_metadata={"filename": payload.filename},
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ])
    await db.commit()

    return {"id": str(document.id), "filename": document.filename, "chunks_created": len(chunks)}


@router.get("")
async def list_documents(current_user: CurrentUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.business_id == current_user.business_id))
    docs = result.scalars().all()
    return [{"id": str(d.id), "filename": d.filename, "status": d.status, "uploaded_at": d.uploaded_at} for d in docs]


@router.delete("/{document_id}")
async def delete_document(document_id: uuid.UUID, current_user: CurrentUser = Depends(get_current_user),
                           db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.business_id == current_user.business_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    await db.delete(document)  # cascades to knowledge_chunks via relationship config
    await db.commit()
    return {"deleted": True}
