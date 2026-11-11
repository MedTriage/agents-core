"""What the agents are told about the patient beyond the current conversation.

Two sources, kept apart and labelled apart:
- the profile, which the patient stated and only the patient edits;
- the record, which the system inferred from earlier chats and the patient never confirmed.

The record never writes the profile. A new inferred fact becomes a suggestion, and only
the patient's click turns it into a profile entry. Where they disagree on safety, the
cautious side wins: an allergy mentioned in a chat is shown to the agents even if the
profile does not list it, marked unconfirmed.
"""
import re
import uuid
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import PRESCRIPTION_BACKGROUND_DAYS
from app.db.models import Checkin, Episode, Prescription, Profile, RecordItem
from app.services import readings
from app.services.followup_content import COMPLAINTS, OVERALL_WORDS

# Scribe record field → record kind. Symptoms, demographics and notes are deliberately
# not carried: last week's cold must not shape today's diagnosis, and age and gender
# come from the profile.
_SCRIBE_FIELDS = {
    "current_conditions": "condition",
    "past_conditions": "condition",
    "medications": "medication",
    "allergies": "allergy",
}
# Record kind → profile list it is accepted into.
PROFILE_LIST = {"condition": "conditions", "medication": "medications", "allergy": "allergies"}

_GENDER_LABEL = {"non_binary": "non-binary", "prefer_not_to_say": "prefer not to say"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def age_on(born: date, today: date) -> int:
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _in_profile(profile: Profile | None, kind: str, norm_text: str) -> bool:
    """Already stated by the patient? A profile entry counts if its name appears as whole
    words in the inferred text, so "metformin 500mg twice daily" matches "Metformin"."""
    if profile is None:
        return False
    for entry in getattr(profile, PROFILE_LIST[kind]) or []:
        name = normalize(entry.get("name", ""))
        if name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", norm_text):
            return True
    return False


def sync_record(db: Session, user_id, conversation_id, scribe_record: dict) -> None:
    """Carry this conversation's long-lived facts into the patient's record. The caller
    commits. Never touches the profile."""
    profile = db.get(Profile, user_id)
    now = datetime.now(timezone.utc)

    for field, kind in _SCRIBE_FIELDS.items():
        for raw in (scribe_record or {}).get(field) or []:
            text = str(raw.get("text") if isinstance(raw, dict) else raw).strip()[:300]
            norm = normalize(text)
            if not norm:
                continue
            item = db.scalar(select(RecordItem).where(
                RecordItem.user_id == user_id, RecordItem.kind == kind, RecordItem.norm_text == norm))
            if item is not None:
                item.last_mentioned_at = now
                continue
            db.add(RecordItem(
                user_id=user_id, conversation_id=conversation_id, kind=kind, text=text,
                norm_text=norm, first_mentioned_at=now, last_mentioned_at=now,
                status="inferred" if _in_profile(profile, kind, norm) else "suggested",
            ))
            db.flush()


def _profile_lines(profile: Profile | None) -> list[str]:
    if profile is None:
        return []
    has_anything = (profile.date_of_birth or profile.gender or profile.allergies
                    or profile.medications or profile.conditions)
    if not has_anything:
        return []

    def listing(entries) -> str:
        return ", ".join(
            f"{e['name']} ({e['detail']})" if e.get("detail") else e["name"] for e in entries
        ) or "none listed"

    age = age_on(profile.date_of_birth, date.today()) if profile.date_of_birth else None
    gender = _GENDER_LABEL.get(profile.gender, profile.gender) if profile.gender else None
    return [
        "Stated by the patient in their account profile:",
        f"- Age: {age if age is not None else 'not provided'}",
        f"- Gender: {gender or 'not provided'}",
        f"- Allergies: {listing(profile.allergies)}",
        f"- Regular medications: {listing(profile.medications)}",
        f"- Long-term conditions: {listing(profile.conditions)}",
    ]


def _prescription_lines(db: Session, user_id) -> list[str]:
    """What doctors issued through MedTriage recently. Listed apart from the profile, which only
    the patient writes. Issued only: a draft is not something the patient was told to take."""
    since = datetime.now(timezone.utc) - timedelta(days=PRESCRIPTION_BACKGROUND_DAYS)
    issued = db.scalars(select(Prescription).where(
        Prescription.patient_id == user_id, Prescription.status == "issued", Prescription.issued_at >= since,
    ).order_by(Prescription.issued_at.desc())).all()
    lines = []
    for rx in issued:
        for med in (rx.final or {}).get("medications") or []:
            lines.append(f"- {med['drug']} {med['dose']}, {med['route']}, {med['frequency']}, for {med['duration']} "
                         f"(issued {readings.on_day(rx.issued_at.date())} by {rx.issued_by_name})")
    return ["Prescribed by a doctor through MedTriage:", *lines] if lines else []


def _unconfirmed_lines(db: Session, user_id, conversation_id) -> list[str]:
    """Inferred items from OTHER conversations that the patient has neither accepted into
    the profile nor dismissed. This conversation's own items are already in its scribe
    record, and labelling them "earlier" would be false."""
    items = db.scalars(
        select(RecordItem)
        .where(RecordItem.user_id == user_id,
               RecordItem.status.in_(("suggested", "inferred")),
               RecordItem.conversation_id.is_distinct_from(conversation_id))
        .order_by(RecordItem.kind, RecordItem.last_mentioned_at.desc())
    ).all()
    if not items:
        return []
    return ["Mentioned in earlier conversations, not confirmed by the patient:"] + [
        f"- {i.kind.capitalize()}: {i.text} (last mentioned {i.last_mentioned_at.date().isoformat()})"
        for i in items
    ]


def _followup_lines(db: Session, conversation_id) -> list[str]:
    """When a follow-up check-in sent this conversation back through triage: what was being
    watched, and what the patient said at the check-in."""
    episode = db.scalar(select(Episode).where(
        Episode.conversation_id == conversation_id, Episode.state == "active",
        Episode.awaiting_retriage.is_(True)))
    if episode is None:
        return []
    checkin = db.scalar(select(Checkin).where(Checkin.episode_id == episode.id, Checkin.answered_at.is_not(None))
                        .order_by(Checkin.answered_at.desc()).limit(1))
    lines = ["Follow-up of this conversation:",
             f"- Watched since {episode.opened_at.date().isoformat()} for: "
             f"{COMPLAINTS[episode.complaint].label.lower()}"]
    if episode.usual_course:
        lines.append(f"- Usual course: {episode.usual_course}")
    if checkin is not None:
        added = f", and added: {checkin.new_text}" if checkin.new_text else ""
        lines.append(f"- At the check-in on {checkin.answered_at.date().isoformat()} the patient said it is "
                     f"{OVERALL_WORDS[checkin.overall]}{added}.")
    return lines


def build_background(db: Session, user_id, conversation_id) -> str | None:
    """The PATIENT BACKGROUND block for every turn, first turn included. None when
    nothing is known, so prompts carry no empty section."""
    uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    sections = [_profile_lines(db.get(Profile, uid)), readings.background_lines(db, uid),
                _prescription_lines(db, uid), _unconfirmed_lines(db, uid, conversation_id),
                _followup_lines(db, conversation_id)]
    return "\n\n".join("\n".join(s) for s in sections if s) or None
