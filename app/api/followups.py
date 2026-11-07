from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.conversations import message_out
from app.api.deps import require_doctor, require_patient
from app.db.models import Episode, Message, User
from app.db.session import get_db
from app.services import followup
from app.services.followup_content import COMPLAINTS

router = APIRouter(tags=["followups"])


class CheckinIn(BaseModel):
    overall: Literal["better", "same", "worse"]
    # Exactly as the episode listed them; anything else is refused.
    signs: list[str] = Field(default_factory=list, max_length=20)
    new: str | None = Field(default=None, max_length=2000)


class PostponeIn(BaseModel):
    hours: int = Field(ge=1, le=168)


def _call(fn, *args):
    try:
        return fn(*args)
    except followup.NotFound:
        raise HTTPException(404, "Not found.")
    except followup.Conflict as conflict:
        raise HTTPException(409, str(conflict))
    except followup.Invalid as invalid:
        raise HTTPException(422, str(invalid))


@router.post("/followups/{episode_id}/checkin")
def checkin(episode_id: str, body: CheckinIn, user: User = Depends(require_patient),
            db: Session = Depends(get_db)):
    result = _call(followup.answer_checkin, db, user, episode_id, body.overall, body.signs, body.new)
    message = db.get(Message, result["message_id"]) if result["message_id"] else None
    episode = db.get(Episode, episode_id)
    return {"outcome": result["outcome"], "message": message_out(message) if message else None,
            "followup": followup.episode_out(db, episode)}


@router.post("/followups/{episode_id}/postpone")
def postpone(episode_id: str, body: PostponeIn, user: User = Depends(require_patient),
             db: Session = Depends(get_db)):
    _call(followup.postpone, db, user, episode_id, body.hours)
    return followup.episode_out(db, db.get(Episode, episode_id))


@router.post("/followups/{episode_id}/stop")
def stop(episode_id: str, user: User = Depends(require_patient), db: Session = Depends(get_db)):
    _call(followup.stop, db, user, episode_id)
    return followup.episode_out(db, db.get(Episode, episode_id))


@router.get("/doctor/followups")
def unanswered(doctor: User = Depends(require_doctor), db: Session = Depends(get_db)):
    """Patients who went silent after a case this doctor approved. Only this doctor's."""
    rows = db.execute(
        select(Episode, User.name).join(User, User.id == Episode.user_id)
        .where(Episode.reviewing_doctor_id == doctor.id, Episode.state == "lapsed")
        .order_by(Episode.closed_at.desc())
    ).all()
    return [
        {"id": str(e.id), "patient_name": name, "label": COMPLAINTS[e.complaint].label,
         "opened_at": e.opened_at.isoformat(), "lapsed_at": e.closed_at.isoformat() if e.closed_at else None}
        for e, name in rows
    ]
