"""One triage turn, from the server's side.

prepare_turn decides whether the turn may run and builds the graph's input from what the
server holds — never from what the client sends. finish_turn persists the outcome. Both
are separate from running the graph so the blocking and streaming endpoints share them,
and so a streamed turn is saved even if the browser has gone away.
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.config import APP_BASE_URL, DOCTOR_DIGEST_MINUTES, TURN_BUSY_MINUTES, TURNS_PER_HOUR
from app.db.models import Conversation, EmailOutbox, Message, Profile, ReviewCase, User
from app.db.session import SessionLocal
from app.graph.nodes.scribe_node import RECORD_KEYS
from app.services.explain import build_explanation
from app.services.outbox import enqueue
from app.services.patient_context import build_background, sync_record

log = logging.getLogger(__name__)

# What a stored assistant message keeps for the audit views. Raw retrieval context is left
# out: it is large, and the branch outputs already carry what was concluded from it.
_PIPELINE_KEYS = ("intent_type", "intent_confidence", "parametric_output", "kgrag_output",
                  "mcp_output", "orchestrator_output", "orchestrator_decision", "guardian_output",
                  "is_emergency")
_LEVELS = ("level_1", "level_2", "level_3")


@dataclass
class Turn:
    """A turn that may run: the graph's input, plus what the review case must record
    about the moment the background was built."""
    state: dict
    profile_updated_at: datetime | None


class TurnRefused(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_id(value) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


def own_conversation(db: Session, user: User, conversation_id) -> Conversation:
    """The caller's conversation, or 404 — someone else's is indistinguishable from none."""
    cid = _parse_id(conversation_id)
    conversation = db.get(Conversation, cid) if cid else None
    if conversation is None or conversation.user_id != user.id:
        raise TurnRefused(404, "Conversation not found.")
    return conversation


def prepare_turn(db: Session, user: User, conversation_id, text: str) -> Turn:
    """Check the turn may run, claim the conversation, and build the graph's input state."""
    hour_ago = _now() - timedelta(hours=1)
    recent = db.scalar(
        select(func.count()).select_from(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Conversation.user_id == user.id, Message.role == "user", Message.created_at > hour_ago)
    )
    if recent >= TURNS_PER_HOUR:
        raise TurnRefused(429, "You've reached the hourly limit. Try again later.")

    if conversation_id:
        conversation = own_conversation(db, user, conversation_id)
        if conversation.locked:
            raise TurnRefused(423, "This conversation was escalated and is closed.")
    else:
        conversation = Conversation(user_id=user.id)
        db.add(conversation)
        db.flush()

    # Claim the conversation atomically. Expired claims count as free, so a crashed turn
    # cannot hold it forever.
    claimed = db.execute(
        update(Conversation)
        .where(Conversation.id == conversation.id,
               (Conversation.busy_until.is_(None)) | (Conversation.busy_until < func.now()))
        .values(busy_until=func.now() + timedelta(minutes=TURN_BUSY_MINUTES))
        .returning(Conversation.id)
    ).first()
    if claimed is None:
        db.rollback()
        raise TurnRefused(409, "A reply in this conversation is still being prepared.")

    history = [
        {"role": m.role, "content": m.content}
        for m in db.scalars(select(Message).where(Message.conversation_id == conversation.id)
                            .order_by(Message.id))
    ]
    profile = db.get(Profile, user.id)
    state = {
        "conversation_id": str(conversation.id),
        "user_input": text,
        "chat_history": history,
        # The scribe's prior record for THIS conversation; None on a first turn.
        "scribe_output": conversation.scribe_record,
        "patient_background": build_background(db, user.id, conversation.id),
    }
    turn = Turn(state=state, profile_updated_at=profile.updated_at if profile else None)
    db.commit()
    return turn


def release(conversation_id) -> None:
    """Free a conversation whose turn failed. Nothing else is saved for a failed turn, and
    a conversation that failed on its first turn is removed rather than left empty."""
    with SessionLocal() as db:
        conversation = db.get(Conversation, _parse_id(conversation_id))
        if conversation is None:
            return
        has_messages = db.scalar(select(Message.id).where(Message.conversation_id == conversation.id).limit(1))
        if has_messages is None:
            db.delete(conversation)
        else:
            conversation.busy_until = None
        db.commit()


def _triage_level(result: dict) -> str:
    """The guardian's level. Missing or malformed fails safe to physician review."""
    level = result.get("triage_level") or (result.get("guardian_output") or {}).get("triage_level")
    return level if level in _LEVELS else "level_2"


def _clean_scribe_record(record) -> dict | None:
    if not isinstance(record, dict):
        return None
    return {k: v for k, v in record.items() if k in RECORD_KEYS or k == "turn"}


def _notify_doctors(db: Session) -> None:
    """At most one "cases waiting" email per doctor per window, however many cases arrive."""
    since = _now() - timedelta(minutes=DOCTOR_DIGEST_MINUTES)
    doctors = db.scalars(select(User).where(
        User.role == "doctor", User.email_verified_at.is_not(None), User.password_hash.is_not(None)))
    for doctor in doctors:
        recently = db.scalar(select(EmailOutbox.id).where(
            EmailOutbox.recipient == doctor.email, EmailOutbox.template == "cases_waiting",
            EmailOutbox.created_at > since))
        if recently is None:
            enqueue(db, doctor.email, "cases_waiting", name=doctor.name, link=f"{APP_BASE_URL}/doctor")


def finish_turn(turn: Turn, result: dict) -> dict:
    """Persist a completed turn and return the payload the client receives."""
    state = turn.state
    level = _triage_level(result)
    response = result.get("orchestrator_response") or (result.get("orchestrator_output") or {}).get("response") or ""
    explanation = build_explanation(result)
    record = _clean_scribe_record(result.get("scribe_output"))

    with SessionLocal() as db:
        conversation = db.get(Conversation, _parse_id(state["conversation_id"]))
        db.add(Message(conversation_id=conversation.id, role="user", content=state["user_input"]))
        assistant = Message(
            conversation_id=conversation.id, role="assistant", content=response, triage_level=level,
            explanation=explanation, pipeline={k: result.get(k) for k in _PIPELINE_KEYS},
        )
        db.add(assistant)
        db.flush()

        if record is not None:
            conversation.scribe_record = record
            sync_record(db, conversation.user_id, conversation.id, record)
        conversation.title = conversation.title or (result.get("title") or None)

        if level == "level_2":
            db.add(ReviewCase(
                message_id=assistant.id, conversation_id=conversation.id, patient_id=conversation.user_id,
                context_snapshot={
                    "background": state.get("patient_background"),
                    "profile_updated_at": (turn.profile_updated_at.isoformat()
                                           if turn.profile_updated_at else None),
                },
            ))
            _notify_doctors(db)
        elif level == "level_3":
            conversation.locked = True

        conversation.busy_until = None
        db.commit()
        message_id = assistant.id

    return {
        **result,
        "triage_level": level,
        "explanation": explanation,
        "conversation_id": state["conversation_id"],
        "message_id": message_id,
    }
