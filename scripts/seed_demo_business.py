"""
Idempotently seeds a synthetic demo tenant and the sample Hinglish support KB.

Run this from a trusted machine against the target database:
    $env:DEMO_EMAIL = "your-dedicated-demo-email@example.com"
    $env:DEMO_PASSWORD = "a-unique-password-of-at-least-12-chars"
    python scripts/seed_demo_business.py

No demo password is committed or printed. The script reuses an existing demo
identity and skips a complete KB; interrupted/partial KB seeds are replaced.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import json
import os
from collections.abc import Mapping

from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy import delete, func, select

from app.db.models import Business, Document, KnowledgeChunk, User
from app.db.session import AsyncSessionLocal, engine
from app.retrieval.chunking import chunk_text

KB_PATH = Path(__file__).parent.parent / "data" / "knowledge_base" / "support_docs.json"
DEMO_FILENAME = "support_docs.json"


def load_demo_credentials(environ: Mapping[str, str] | None = None) -> tuple[str, str, str]:
    """Read and validate demo credentials without ever supplying a baked-in password."""
    values = os.environ if environ is None else environ
    raw_email = (values.get("DEMO_EMAIL") or "").strip()
    password = values.get("DEMO_PASSWORD") or ""
    business_name = (values.get("DEMO_BUSINESS_NAME") or "Demo E-Commerce Co").strip()
    if not raw_email or not password:
        raise RuntimeError("Set DEMO_EMAIL and DEMO_PASSWORD before running the demo seed script")
    try:
        email = str(TypeAdapter(EmailStr).validate_python(raw_email)).lower()
    except ValidationError as exc:
        raise ValueError("DEMO_EMAIL must be a valid email address") from exc
    if len(password) < 12:
        raise ValueError("DEMO_PASSWORD must contain at least 12 characters")
    if len(password.encode("utf-8")) > 72:
        raise ValueError("DEMO_PASSWORD must be at most 72 UTF-8 bytes (bcrypt limit)")
    if not business_name or len(business_name) > 200:
        raise ValueError("DEMO_BUSINESS_NAME must contain 1-200 characters")
    return email, password, business_name


async def _get_or_create_demo_identity(db, email: str, password: str, business_name: str):
    from app.core.security import hash_password

    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is not None:
        business = (await db.execute(select(Business).where(Business.id == user.business_id))).scalar_one()
        if business.name != business_name:
            raise RuntimeError(
                "DEMO_EMAIL already belongs to a different business; choose a dedicated demo email"
            )
        return business, user, False

    business = Business(name=business_name, industry="retail")
    user = User(
        business=business,
        email=email,
        hashed_password=hash_password(password),
        role="admin",
    )
    db.add_all([business, user])
    await db.commit()
    await db.refresh(business)
    await db.refresh(user)
    return business, user, True


async def _seed_knowledge_base(db, business_id) -> tuple[int, int, bool]:
    from app.retrieval.retriever import upsert_chunk

    docs = json.loads(KB_PATH.read_text(encoding="utf-8"))
    expected_chunks = sum(len(chunk_text(entry["text"])) for entry in docs)

    existing_docs = (await db.execute(
        select(Document).where(
            Document.business_id == business_id,
            Document.filename == DEMO_FILENAME,
        ).order_by(Document.uploaded_at.desc())
    )).scalars().all()

    complete_doc = None
    for candidate in existing_docs:
        count = (await db.execute(
            select(func.count()).select_from(KnowledgeChunk).where(
                KnowledgeChunk.document_id == candidate.id
            )
        )).scalar_one()
        if candidate.status == "ready" and count == expected_chunks:
            complete_doc = candidate
            break

    if complete_doc is not None:
        # Remove any stale duplicates left by an older non-idempotent seed run.
        for candidate in existing_docs:
            if candidate.id != complete_doc.id:
                await db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == candidate.id))
                await db.delete(candidate)
        await db.commit()
        return len(docs), expected_chunks, False

    # A previous attempt may have stopped halfway through embedding the KB.
    for candidate in existing_docs:
        await db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == candidate.id))
        await db.delete(candidate)
    if existing_docs:
        await db.commit()

    document = Document(
        business_id=business_id,
        filename=DEMO_FILENAME,
        file_type="json",
        status="processing",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    document_id = document.id
    written = 0
    try:
        for entry in docs:
            for chunk in chunk_text(entry["text"]):
                await upsert_chunk(db, document.id, business_id, chunk, {"category": entry["category"]})
                written += 1
        document.status = "ready"
        await db.commit()
    except Exception:
        await db.rollback()
        failed_doc = await db.get(Document, document_id)
        if failed_doc is not None:
            failed_doc.status = "failed"
            await db.commit()
        raise

    return len(docs), written, True


async def main():
    email, password, business_name = load_demo_credentials()
    async with AsyncSessionLocal() as db:
        business, _user, created = await _get_or_create_demo_identity(
            db, email, password, business_name
        )
        source_count, chunk_count, kb_created = await _seed_knowledge_base(db, business.id)
        print(f"Demo business: {business.id}")
        print(f"Demo login email: {email}")
        print("Demo identity: created" if created else "Demo identity: reused")
        print(f"Knowledge base: {source_count} source entries, {chunk_count} chunks")
        print("Knowledge base: seeded" if kb_created else "Knowledge base: already complete; skipped")
        print("Password was not printed. Keep it private and share it only with intended interviewers.")


if __name__ == "__main__":
    async def run():
        try:
            await main()
        finally:
            await engine.dispose()

    asyncio.run(run())
