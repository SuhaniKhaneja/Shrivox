"""
Seeds one demo business, an admin user, and the sample Hinglish support
KB -- enough to log in and exercise the full multi-tenant flow locally.

    python scripts/seed_demo_business.py

Why a seed script instead of "just use the /auth/register + /documents/
upload API calls":
Both paths are valid and exercise the same code either way -- this
script is just faster for local dev/demo setup (one command, no need
to have the server running yet) since it writes directly via the ORM.
"""

import asyncio
import json
import uuid
from pathlib import Path

from app.db.session import AsyncSessionLocal
from app.db.models import Business, User, Document
from app.core.security import hash_password
from app.retrieval.retriever import upsert_chunk
from app.retrieval.chunking import chunk_text

KB_PATH = Path(__file__).parent.parent / "data" / "knowledge_base" / "support_docs.json"
# Must pass pydantic EmailStr in /auth/login: reserved TLDs such as .test are rejected (422).
DEMO_EMAIL = "admin@example.com"
DEMO_PASSWORD = "demo_password_123"


async def main():
    async with AsyncSessionLocal() as db:
        business = Business(id=uuid.uuid4(), name="Demo E-Commerce Co", industry="retail")
        db.add(business)
        await db.flush()

        user = User(
            id=uuid.uuid4(), business_id=business.id, email=DEMO_EMAIL,
            hashed_password=hash_password(DEMO_PASSWORD), role="admin",
        )
        db.add(user)
        await db.commit()

        docs = json.loads(KB_PATH.read_text())
        document = Document(
            id=uuid.uuid4(), business_id=business.id,
            filename="support_docs.json", file_type="json", status="processing",
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)

        total_chunks = 0
        for entry in docs:
            for chunk in chunk_text(entry["text"]):
                await upsert_chunk(db, document.id, business.id, chunk, {"category": entry["category"]})
                total_chunks += 1

        document.status = "ready"
        await db.commit()

        print(f"Business created: {business.id}")
        print(f"Login: {DEMO_EMAIL} / {DEMO_PASSWORD}")
        print(f"Knowledge base seeded: {total_chunks} chunks from {len(docs)} source entries")


if __name__ == "__main__":
    asyncio.run(main())
