"""Putting an answer in front of a doctor. Shared by triage turns and follow-up check-ins."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import APP_BASE_URL, DOCTOR_DIGEST_MINUTES
from app.db.models import EmailOutbox, Message, ReviewCase, User
from app.services.outbox import enqueue


def notify_doctors(db: Session) -> None:
    """At most one "cases waiting" email per doctor per window, however many cases arrive."""
    since = datetime.now(timezone.utc) - timedelta(minutes=DOCTOR_DIGEST_MINUTES)
    doctors = db.scalars(select(User).where(
        User.role == "doctor", User.email_verified_at.is_not(None), User.password_hash.is_not(None)))
    for doctor in doctors:
        recently = db.scalar(select(EmailOutbox.id).where(
            EmailOutbox.recipient == doctor.email, EmailOutbox.template == "cases_waiting",
            EmailOutbox.created_at > since))
        if recently is None:
            enqueue(db, doctor.email, "cases_waiting", name=doctor.name, link=f"{APP_BASE_URL}/doctor")


def open_case(db: Session, message: Message, patient_id, background: str | None,
              profile_updated_at: datetime | None) -> ReviewCase:
    """A review case for an assistant message, recording the background the answer was
    given, and a heads-up to the doctors. The caller commits."""
    case = ReviewCase(
        message_id=message.id, conversation_id=message.conversation_id, patient_id=patient_id,
        context_snapshot={
            "background": background,
            "profile_updated_at": profile_updated_at.isoformat() if profile_updated_at else None,
        },
    )
    db.add(case)
    db.flush()
    notify_doctors(db)
    return case


def case_question(db: Session, case: ReviewCase) -> str:
    """The patient message the case's answer replied to."""
    return db.scalar(
        select(Message.content)
        .where(Message.conversation_id == case.conversation_id, Message.role == "user",
               Message.id < case.message_id)
        .order_by(Message.id.desc()).limit(1)
    ) or ""
