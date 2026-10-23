from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.auth import user_out
from app.api.conversations import message_out
from app.api.deps import current_user, require_patient
from app.db.models import (
    GENDERS, Conversation, EmailOutbox, LoginAttempt, Message, Profile, RecordItem, ReviewCase, User,
)
from app.db.session import get_db
from app.services.outbox import enqueue
from app.services.patient_context import PROFILE_LIST
from app.services.security import verify_password

router = APIRouter(prefix="/me", tags=["me"])


class ProfileEntry(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    detail: str | None = Field(default=None, max_length=300)


class ProfileIn(BaseModel):
    date_of_birth: date | None = None
    gender: str | None = None
    allergies: list[ProfileEntry] = Field(default_factory=list, max_length=100)
    medications: list[ProfileEntry] = Field(default_factory=list, max_length=100)
    conditions: list[ProfileEntry] = Field(default_factory=list, max_length=100)

    @field_validator("gender")
    @classmethod
    def _known_gender(cls, value):
        if value is not None and value not in GENDERS:
            raise ValueError(f"gender must be one of {', '.join(GENDERS)}")
        return value

    @field_validator("date_of_birth")
    @classmethod
    def _not_in_future(cls, value):
        if value is not None and value > date.today():
            raise ValueError("date of birth cannot be in the future")
        return value


def profile_out(profile: Profile) -> dict:
    return {
        "date_of_birth": profile.date_of_birth.isoformat() if profile.date_of_birth else None,
        "gender": profile.gender,
        "allergies": profile.allergies or [],
        "medications": profile.medications or [],
        "conditions": profile.conditions or [],
    }


def _profile(db: Session, user: User) -> Profile:
    profile = db.get(Profile, user.id)
    if profile is None:
        profile = Profile(user_id=user.id, allergies=[], medications=[], conditions=[])
        db.add(profile)
        db.flush()
    return profile


class DeleteIn(BaseModel):
    password: str = Field(max_length=256)


@router.get("")
def me(user: User = Depends(current_user)):
    return user_out(user)


@router.delete("", status_code=204)
def delete_account(body: DeleteIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Delete the account and everything it owns. The database cascades through profile,
    sessions, conversations, messages, record and review cases; the access log keeps the
    action and time but loses the patient. Rows keyed only by the email address go too."""
    if not verify_password(user.password_hash, body.password):
        raise HTTPException(403, "Incorrect password.")
    email, name = user.email, user.name
    db.delete(user)
    db.execute(delete(LoginAttempt).where(LoginAttempt.email == email))
    db.execute(delete(EmailOutbox).where(EmailOutbox.recipient == email))
    enqueue(db, email, "account_deleted", name=name)
    db.commit()
    return Response(status_code=204)


@router.get("/profile")
def get_profile(user: User = Depends(require_patient), db: Session = Depends(get_db)):
    return profile_out(_profile(db, user))


@router.put("/profile")
def put_profile(body: ProfileIn, user: User = Depends(require_patient), db: Session = Depends(get_db)):
    profile = _profile(db, user)
    profile.date_of_birth = body.date_of_birth
    profile.gender = body.gender
    for field in ("allergies", "medications", "conditions"):
        setattr(profile, field, [e.model_dump() for e in getattr(body, field)])
    db.commit()
    return profile_out(profile)


@router.get("/export")
def export(user: User = Depends(require_patient), db: Session = Depends(get_db)):
    conversations = db.scalars(select(Conversation).where(Conversation.user_id == user.id)
                               .order_by(Conversation.created_at)).all()
    cases = {c.message_id: c for c in db.scalars(select(ReviewCase).where(ReviewCase.patient_id == user.id))}
    record = db.scalars(select(RecordItem).where(RecordItem.user_id == user.id)
                        .order_by(RecordItem.first_mentioned_at)).all()
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "account": user_out(user),
        "profile": profile_out(_profile(db, user)),
        "record": [{"kind": i.kind, "text": i.text, "status": i.status,
                    "first_mentioned_at": i.first_mentioned_at.isoformat(),
                    "last_mentioned_at": i.last_mentioned_at.isoformat()} for i in record],
        "conversations": [
            {"id": str(c.id), "title": c.title, "created_at": c.created_at.isoformat(),
             "messages": [message_out(m, cases.get(m.id)) for m in db.scalars(
                 select(Message).where(Message.conversation_id == c.id).order_by(Message.id))]}
            for c in conversations
        ],
    }


@router.get("/suggestions")
def suggestions(user: User = Depends(require_patient), db: Session = Depends(get_db)):
    items = db.scalars(
        select(RecordItem)
        .where(RecordItem.user_id == user.id, RecordItem.status == "suggested")
        .order_by(RecordItem.first_mentioned_at)
    ).all()
    return [
        {"id": i.id, "kind": i.kind, "text": i.text,
         "first_mentioned_at": i.first_mentioned_at.isoformat()}
        for i in items
    ]


def _own_suggestion(db: Session, user: User, item_id: int) -> RecordItem:
    item = db.get(RecordItem, item_id)
    # Someone else's item is indistinguishable from a missing one.
    if item is None or item.user_id != user.id or item.status != "suggested":
        raise HTTPException(404, "No such suggestion.")
    return item


@router.post("/suggestions/{item_id}/accept")
def accept_suggestion(item_id: int, user: User = Depends(require_patient), db: Session = Depends(get_db)):
    item = _own_suggestion(db, user, item_id)
    profile = _profile(db, user)
    field = PROFILE_LIST[item.kind]
    setattr(profile, field, [*(getattr(profile, field) or []), {"name": item.text, "detail": None}])
    item.status = "accepted"
    db.commit()
    return profile_out(profile)


@router.post("/suggestions/{item_id}/dismiss")
def dismiss_suggestion(item_id: int, user: User = Depends(require_patient), db: Session = Depends(get_db)):
    item = _own_suggestion(db, user, item_id)
    item.status = "dismissed"
    db.commit()
    return {"status": "dismissed"}
