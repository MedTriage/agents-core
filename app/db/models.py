"""Persistent state: accounts, the patient's profile and record, conversations, the review
queue, and the email outbox. Schema changes go through Alembic (migrations/)."""
import uuid
from datetime import date, datetime

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Date, DateTime, ForeignKey, Integer, String, Text,
    UniqueConstraint, Uuid, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

ROLES = ("patient", "doctor")
GENDERS = ("female", "male", "non_binary", "other", "prefer_not_to_say")
TOKEN_PURPOSES = ("verify", "reset", "invite")
RECORD_KINDS = ("condition", "medication", "allergy")
RECORD_STATUSES = ("inferred", "suggested", "accepted", "dismissed")
CASE_STATUSES = ("pending", "verified", "rejected")


def _in(column: str, values) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_in("role", ROLES), name="role_valid"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Stored lower-cased, so uniqueness is case-insensitive.
    email: Mapped[str] = mapped_column(String(320), unique=True)
    # Null until an invited doctor accepts.
    password_hash: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16))
    name: Mapped[str] = mapped_column(String(200))
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # sha256 of the token; the token itself exists only in the client's cookie.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmailToken(Base):
    __tablename__ = "email_tokens"
    __table_args__ = (CheckConstraint(_in("purpose", TOKEN_PURPOSES), name="purpose_valid"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    purpose: Mapped[str] = mapped_column(String(16))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LoginAttempt(Base):
    __tablename__ = "login_attempts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    ip: Mapped[str] = mapped_column(String(64), index=True)
    success: Mapped[bool] = mapped_column(Boolean)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class Profile(Base):
    """What the patient states about themselves. Only the patient writes it — never the
    model, which is why inferred facts live in RecordItem and only become profile entries
    when the patient accepts them."""
    __tablename__ = "profiles"
    __table_args__ = (CheckConstraint(f"gender IS NULL OR {_in('gender', GENDERS)}", name="gender_valid"),)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    gender: Mapped[str | None] = mapped_column(String(32))
    # Each a list of {"name": str, "detail": str | null}.
    allergies: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    medications: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    conditions: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # Set by a level_3 turn; the server refuses further turns, not just the UI.
    locked: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    # One turn at a time. Expires so a crashed turn cannot hold the conversation forever.
    busy_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The scribe's per-conversation record, in exactly the shape the scribe node uses.
    scribe_record: Mapped[dict | None] = mapped_column(JSONB)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (CheckConstraint(_in("role", ("user", "assistant")), name="role_valid"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    triage_level: Mapped[str | None] = mapped_column(String(16))
    # Patient-safe explanation (explain.py) and the doctor-facing audit data.
    explanation: Mapped[dict | None] = mapped_column(JSONB)
    pipeline: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RecordItem(Base):
    """A long-lived fact the system inferred from a chat. Symptoms are deliberately absent:
    last week's cold must not shape today's diagnosis."""
    __tablename__ = "record_items"
    __table_args__ = (
        UniqueConstraint("user_id", "kind", "norm_text", name="record_item_unique"),
        CheckConstraint(_in("kind", RECORD_KINDS), name="kind_valid"),
        CheckConstraint(_in("status", RECORD_STATUSES), name="status_valid"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL")
    )
    kind: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(String(300))
    norm_text: Mapped[str] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(16))
    first_mentioned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_mentioned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReviewCase(Base):
    __tablename__ = "review_cases"
    __table_args__ = (CheckConstraint(_in("status", CASE_STATUSES), name="status_valid"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), unique=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", server_default="pending")
    doctor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    notes: Mapped[str | None] = mapped_column(Text)
    # The patient background exactly as the agents saw it when they answered.
    context_snapshot: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AccessLog(Base):
    __tablename__ = "access_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(32))
    case_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("review_cases.id", ondelete="SET NULL"))
    # Nulled when the patient deletes their account; the action and time are kept.
    patient_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmailOutbox(Base):
    """Emails are written here in the same transaction as the event that caused them, and
    sent by a background loop — so a slow or failing SMTP server never fails a request."""
    __tablename__ = "email_outbox"
    __table_args__ = (CheckConstraint(_in("status", ("pending", "sent", "failed")), name="status_valid"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    recipient: Mapped[str] = mapped_column(String(320), index=True)
    template: Mapped[str] = mapped_column(String(64))
    # Rendering inputs only: names and links, never health information.
    payload: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="pending", server_default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
