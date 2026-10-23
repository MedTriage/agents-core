from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import require_patient
from app.db.models import Conversation, Message, ReviewCase, User
from app.db.session import get_db
from app.services.turns import TurnRefused, own_conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _own(db: Session, user: User, conversation_id: str) -> Conversation:
    try:
        return own_conversation(db, user, conversation_id)
    except TurnRefused as refused:
        raise HTTPException(refused.status, refused.detail)


def review_out(case: ReviewCase | None) -> dict | None:
    if case is None:
        return None
    return {"status": case.status, "notes": case.notes,
            "reviewed_at": case.reviewed_at.isoformat() if case.reviewed_at else None}


def message_out(message: Message, case: ReviewCase | None = None) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "triage_level": message.triage_level,
        "explanation": message.explanation,
        "pipeline": message.pipeline,
        "created_at": message.created_at.isoformat(),
        "review": review_out(case),
    }


@router.get("")
def list_conversations(user: User = Depends(require_patient), db: Session = Depends(get_db)):
    # level_N sorts in risk order as a string, so max() is the highest level reached.
    summary = (
        select(Message.conversation_id, func.count().label("n"),
               func.max(Message.triage_level).label("highest"))
        .group_by(Message.conversation_id).subquery()
    )
    rows = db.execute(
        select(Conversation, summary.c.n, summary.c.highest)
        .outerjoin(summary, summary.c.conversation_id == Conversation.id)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    ).all()
    return [
        {"id": str(c.id), "title": c.title, "locked": c.locked,
         "created_at": c.created_at.isoformat(), "updated_at": c.updated_at.isoformat(),
         "message_count": n or 0, "highest_triage_level": highest}
        for c, n, highest in rows
    ]


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, user: User = Depends(require_patient),
                     db: Session = Depends(get_db)):
    conversation = _own(db, user, conversation_id)
    messages = db.scalars(select(Message).where(Message.conversation_id == conversation.id)
                          .order_by(Message.id)).all()
    cases = {c.message_id: c for c in db.scalars(
        select(ReviewCase).where(ReviewCase.conversation_id == conversation.id))}
    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "locked": conversation.locked,
        "messages": [message_out(m, cases.get(m.id)) for m in messages],
    }


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, user: User = Depends(require_patient),
                        db: Session = Depends(get_db)):
    db.delete(_own(db, user, conversation_id))
    db.commit()
    return Response(status_code=204)
