"""
Multi-tenant Postgres schema.

Why multi-tenant at the schema level (business_id on every tenant-owned
table) rather than one deployment per customer:
A real support-AI product serves many businesses off one deployment --
that's the whole value proposition (shared infra, per-tenant config
and knowledge base). The critical schema rule this creates: every
query that touches tenant data MUST filter by business_id, especially
in retrieval (app/retrieval/retriever.py) -- a missing business_id
filter there is a real data-leak bug class (Business A's customer
gets an answer sourced from Business B's private docs), not a
hypothetical. That's why business_id sits on knowledge_chunks,
documents, calls, and evaluations directly rather than being inferred
via a join every time.

Why pgvector instead of a separate vector DB:
Postgres already holds every other tenant table here. Running vector
search in the same database means one fewer moving part, one
transactional boundary (a document delete and its embedding rows can
be a single transaction), and it's a deliberately defensible "we
didn't reach for a dedicated vector store on day one" position --
migrate to a dedicated store later if concurrent write load on
Postgres becomes the bottleneck, not before.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    String, Text, Integer, Float, ForeignKey, DateTime, Boolean,
    CheckConstraint, ARRAY, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

EMBEDDING_DIM = 1536  # text-embedding-3-small


class Base(DeclarativeBase):
    pass


class Business(Base):
    """A tenant. Every other tenant-owned table below references this."""
    __tablename__ = "businesses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    industry: Mapped[str | None] = mapped_column(String(100), nullable=True)
    voice_config: Mapped[dict] = mapped_column(JSONB, default=dict)       # TTS voice id, tone
    escalation_rules: Mapped[dict] = mapped_column(JSONB, default=dict)   # keywords, thresholds
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    users: Mapped[list["User"]] = relationship(back_populates="business", cascade="all, delete-orphan")


class User(Base):
    """Dashboard login -- a business admin/agent, not an end customer."""
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("businesses.id"))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    business: Mapped["Business"] = relationship(back_populates="users")

    __table_args__ = (CheckConstraint("role IN ('admin','agent','viewer')", name="ck_user_role"),)


class Call(Base):
    """One row per customer interaction/session (was `Conversation` in Phase 1)."""
    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("businesses.id"), index=True)
    channel: Mapped[str] = mapped_column(String(20))  # "browser" | "phone"
    caller_ref: Mapped[str | None] = mapped_column(String(50), nullable=True)  # phone number or session token
    status: Mapped[str] = mapped_column(String(20), default="active")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    csat_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recording_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    messages: Mapped[list["Message"]] = relationship(back_populates="call", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("channel IN ('browser','phone')", name="ck_call_channel"),
        CheckConstraint("status IN ('active','completed','escalated','failed')", name="ck_call_status"),
    )


class Message(Base):
    """Every turn within a call. Per-turn (not per-call) logging is what
    makes the evaluation pipeline possible -- see app/evaluation."""
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id"), index=True)
    role: Mapped[str] = mapped_column(String(10))  # "customer" | "agent" | "system"
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    retrieved_doc_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)), nullable=True)
    safety_flag: Mapped[str | None] = mapped_column(String(30), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    call: Mapped["Call"] = relationship(back_populates="messages")

    __table_args__ = (CheckConstraint("role IN ('customer','agent','system')", name="ck_message_role"),)


class Document(Base):
    """An uploaded knowledge-base source file, owned by a business."""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("businesses.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="processing")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chunks: Mapped[list["KnowledgeChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (CheckConstraint("status IN ('processing','ready','failed')", name="ck_document_status"),)


class KnowledgeChunk(Base):
    """
    Embedded chunk of a document, searched via pgvector.

    business_id is duplicated here (not just reachable via document_id
    join) so the retrieval query can filter WHERE business_id = :bid
    directly in the vector search -- filtering after the fact, or only
    at the application layer, is exactly the kind of bug that leaks
    one tenant's data into another's answer.
    """
    __tablename__ = "knowledge_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"), index=True)
    business_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("businesses.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Prompt(Base):
    """Versioned prompts -- enables A/B comparison and rollback without a deploy."""
    __tablename__ = "prompts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    version: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_prompt_name_version"),)


class Evaluation(Base):
    """One row per eval run against the golden dataset (see app/evaluation)."""
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    business_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("businesses.id"), index=True)
    prompt_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("prompts.id"), nullable=True)
    intent_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    safety_violation_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    run_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
