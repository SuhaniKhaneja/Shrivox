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
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import Document
from app.retrieval.chunking import chunk_text
from app.retrieval.retriever import upsert_chunk
from app.api.deps import get_current_user, CurrentUser

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class DocumentUploadRequest(BaseModel):
    filename: str
    text: str


@router.post("/upload", status_code=201)
async def upload_document(payload: DocumentUploadRequest, current_user: CurrentUser = Depends(get_current_user),
                           db: AsyncSession = Depends(get_db)):
    document = Document(
        id=uuid.uuid4(), business_id=current_user.business_id,
        filename=payload.filename, file_type="text", status="processing",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    chunks = chunk_text(payload.text)
    for chunk in chunks:
        await upsert_chunk(db, document.id, current_user.business_id, chunk, {"filename": payload.filename})

    document.status = "ready"
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
