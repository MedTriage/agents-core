"""The doctor review queue. A doctor sees pending cases and the cases they reviewed —
nothing else, not even another case from the same patient — and every case view and
decision is written to the access log."""
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import require_doctor
from app.api.conversations import message_out
from app.api.me import profile_out
from app.config import APP_BASE_URL
from app.db.models import AccessLog, Conversation, Message, Profile, ReviewCase, User
from app.db.session import get_db
from app.services.outbox import enqueue

router = APIRouter(prefix="/doctor", tags=["doctor"])
public = APIRouter(tags=["status"])


class ReviewIn(BaseModel):
    status: Literal["verified", "rejected"]
    notes: str | None = Field(default=None, max_length=4000)


def _visible(doctor: User):
    return or_(ReviewCase.status == "pending", ReviewCase.doctor_id == doctor.id)


def _case(db: Session, doctor: User, case_id: str) -> ReviewCase:
    case = db.scalar(select(ReviewCase).where(ReviewCase.id == _uuid(case_id), _visible(doctor)))
    if case is None:
        raise HTTPException(404, "Case not found.")
    return case


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(404, "Case not found.")


def _question(db: Session, case: ReviewCase) -> str:
    """The patient message the case's answer replied to."""
    return db.scalar(
        select(Message.content)
        .where(Message.conversation_id == case.conversation_id, Message.role == "user",
               Message.id < case.message_id)
        .order_by(Message.id.desc()).limit(1)
    ) or ""


def _log(db: Session, doctor: User, action: str, case: ReviewCase) -> None:
    db.add(AccessLog(actor_id=doctor.id, action=action, case_id=case.id, patient_id=case.patient_id))


@router.get("/cases")
def list_cases(doctor: User = Depends(require_doctor), db: Session = Depends(get_db)):
    cases = db.scalars(
        select(ReviewCase).where(_visible(doctor))
        .order_by((ReviewCase.status != "pending"), ReviewCase.created_at)
    ).all()
    names = dict(db.execute(select(User.id, User.name).where(
        User.id.in_({c.patient_id for c in cases}))).all()) if cases else {}
    return [
        {"id": str(c.id), "status": c.status, "created_at": c.created_at.isoformat(),
         "reviewed_at": c.reviewed_at.isoformat() if c.reviewed_at else None,
         "patient_name": names.get(c.patient_id), "question": _question(db, c)}
        for c in cases
    ]


@router.get("/cases/{case_id}")
def get_case(case_id: str, doctor: User = Depends(require_doctor), db: Session = Depends(get_db)):
    case = _case(db, doctor, case_id)
    transcript = db.scalars(
        select(Message).where(Message.conversation_id == case.conversation_id,
                              Message.id <= case.message_id).order_by(Message.id)
    ).all()
    patient = db.get(User, case.patient_id)
    profile = db.get(Profile, case.patient_id)
    snapshot_at = (case.context_snapshot or {}).get("profile_updated_at")
    changed = bool(profile and snapshot_at
                   and profile.updated_at > datetime.fromisoformat(snapshot_at))

    _log(db, doctor, "view_case", case)
    db.commit()
    return {
        "id": str(case.id),
        "status": case.status,
        "notes": case.notes,
        "created_at": case.created_at.isoformat(),
        "reviewed_at": case.reviewed_at.isoformat() if case.reviewed_at else None,
        "patient_name": patient.name if patient else None,
        "title": db.scalar(select(Conversation.title).where(Conversation.id == case.conversation_id)),
        "transcript": [message_out(m) for m in transcript],
        # What the agents were told about the patient when they answered.
        "background": (case.context_snapshot or {}).get("background"),
        "profile": profile_out(profile) if profile else None,
        "profile_changed_since": changed,
    }


@router.patch("/cases/{case_id}")
def review_case(case_id: str, body: ReviewIn, doctor: User = Depends(require_doctor),
                db: Session = Depends(get_db)):
    case = _case(db, doctor, case_id)
    if case.status != "pending":
        raise HTTPException(409, "This case has already been reviewed.")

    case.status = body.status
    case.notes = body.notes
    case.doctor_id = doctor.id
    case.reviewed_at = datetime.now(timezone.utc)
    _log(db, doctor, "review_case", case)

    patient = db.get(User, case.patient_id)
    # Says only that a review happened. The decision and the notes are read after
    # signing in, never in the inbox.
    enqueue(db, patient.email, "case_reviewed", name=patient.name,
            link=f"{APP_BASE_URL}/chat?c={case.conversation_id}")
    db.commit()
    return {"id": str(case.id), "status": case.status, "notes": case.notes,
            "reviewed_at": case.reviewed_at.isoformat()}


@public.get("/status")
def status(db: Session = Depends(get_db)):
    """Public header strip: one number, nothing about any case."""
    waiting = db.scalar(select(func.count()).select_from(ReviewCase).where(ReviewCase.status == "pending"))
    return {"waiting": waiting}
