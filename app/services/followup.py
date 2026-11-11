"""Follow-up: an answer becomes a watched episode, checked on at NICE's timing.

Code decides everything here except one thing: Jev names the complaint as an episode opens,
with the word list as its fallback. Three rules carry the safety:
- A check-in can only raise the level of care, never lower it.
- Silence is never recovery: one reminder, then the episode is `lapsed`, and a doctor who
  approved the case is told.
- Emails only say a check-in is waiting; what is being watched stays in the app.
"""
import logging
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import (
    APP_BASE_URL, FOLLOWUP_DEMO_SECONDS, FOLLOWUP_JEV_MIN_PROBABILITY, FOLLOWUP_LAPSE_HOURS,
    FOLLOWUP_QUIET_HOURS, FOLLOWUP_REMINDER_HOURS, FOLLOWUP_TIMEZONE,
)
from app.db.models import Checkin, Conversation, Episode, Message, Profile, User
from app.db.session import SessionLocal
from app.services import jev
from app.services.followup_content import COMPLAINTS, OVERALL_WORDS, classify, plan_for
from app.services.outbox import enqueue
from app.services.patient_context import age_on, build_background
from app.services.review import open_case

log = logging.getLogger(__name__)

_RANK = {"level_1": 1, "level_2": 2, "level_3": 3}
_TZ = ZoneInfo(FOLLOWUP_TIMEZONE)


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


class Invalid(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _demo() -> bool:
    return FOLLOWUP_DEMO_SECONDS > 0


def _wait(*, days: float = 0, hours: float = 0) -> timedelta:
    """A real wait, or the demo toggle's few seconds."""
    return timedelta(seconds=FOLLOWUP_DEMO_SECONDS) if _demo() else timedelta(days=days, hours=hours)


def _quiet(moment: datetime) -> bool:
    start, end = FOLLOWUP_QUIET_HOURS
    hour = moment.astimezone(_TZ).hour
    return hour >= start or hour < end


def _day(moment: datetime) -> str:
    return moment.astimezone(_TZ).strftime("%A %-d %B")


# ── episodes ─────────────────────────────────────────────────────────────────────────

def active_episode(db: Session, conversation_id) -> Episode | None:
    return db.scalar(select(Episode).where(Episode.conversation_id == conversation_id,
                                           Episode.state == "active"))


def latest_episode(db: Session, conversation_id) -> Episode | None:
    return db.scalar(select(Episode).where(Episode.conversation_id == conversation_id)
                     .order_by(Episode.opened_at.desc()).limit(1))


def open_checkin(db: Session, episode: Episode) -> Checkin | None:
    return db.scalar(select(Checkin).where(Checkin.episode_id == episode.id, Checkin.outcome.is_(None)))


def _last_answer(db: Session, episode: Episode) -> Checkin | None:
    return db.scalar(select(Checkin).where(Checkin.episode_id == episode.id, Checkin.answered_at.is_not(None))
                     .order_by(Checkin.answered_at.desc()).limit(1))


def episode_out(db: Session, episode: Episode | None) -> dict | None:
    if episode is None:
        return None
    return {
        "id": str(episode.id),
        "state": episode.state,
        "complaint": episode.complaint,
        "label": COMPLAINTS[episode.complaint].label,
        "usual_course": episode.usual_course,
        "return_signs": episode.return_signs,
        "next_checkin_at": episode.next_checkin_at.isoformat() if episode.next_checkin_at else None,
        "checkin_due": episode.state == "active" and open_checkin(db, episode) is not None,
        "opened_at": episode.opened_at.isoformat() if episode.opened_at else None,
    }


def calendar_out(db: Session, episode: Episode) -> dict:
    """The episode with its dates, for the patient's calendar. Only the next check-in has a fixed
    date: each later one counts from when the one before it is answered, so those are "expected".
    A check-in nobody answered is "missed", never counted as an answer."""
    entries = []
    due_at = None
    for checkin in db.scalars(select(Checkin).where(Checkin.episode_id == episode.id).order_by(Checkin.id)):
        if checkin.answered_at is not None:
            entries.append({"kind": "answered", "at": checkin.answered_at.isoformat(), "overall": checkin.overall})
        elif checkin.outcome == "lapsed":
            entries.append({"kind": "missed", "at": checkin.opened_at.isoformat()})
        elif checkin.outcome is None:
            due_at = checkin.opened_at
            entries.append({"kind": "due", "at": due_at.isoformat()})
    if episode.state == "active":
        if episode.next_checkin_at is not None:
            entries.append({"kind": "upcoming", "at": episode.next_checkin_at.isoformat()})
        moment = episode.next_checkin_at or due_at
        days = episode.checkin_days
        for number in range(episode.checkins_done + 1, len(days)):
            if moment is None:
                break
            moment += _wait(days=days[number] - days[number - 1])
            entries.append({"kind": "expected", "at": moment.isoformat()})
    return {**episode_out(db, episode), "conversation_id": str(episode.conversation_id), "entries": entries}


_JEV_INSTRUCTIONS = ("Which of these illnesses is the patient being treated for? Judge from the "
                     "diagnosis first, then their current conditions and symptoms.")
_JEV_CRITERIA = {
    "sore_throat": "Sore throat: pharyngitis, tonsillitis or a throat infection",
    "ear_infection": "Ear infection: acute otitis media, or earache",
    "sinusitis": "Sinusitis: a sinus infection",
    "cough": "Acute cough: bronchitis or a chest infection",
    "uti": "Lower urinary tract infection: cystitis or a bladder infection",
    "general": "None of these: anything else",
}


def _complaint(diagnosis: str | None, record: dict | None) -> str:
    """The complaint to watch. Read from the diagnosis and the scribe's record (the whole
    conversation, already understood) rather than the raw last message, where "no cough"
    reads as a cough. Jev decides when it is sure, the general plan when it isn't, and the
    word list when it can't be reached."""
    record = record or {}
    conditions = list(record.get("current_conditions") or [])
    symptoms = [s["text"] if isinstance(s, dict) else s for s in record.get("symptoms") or []]
    try:
        probabilities = jev.choose({"diagnosis": diagnosis or "", "conditions": conditions,
                                    "symptoms": symptoms}, _JEV_INSTRUCTIONS, _JEV_CRITERIA)
    except Exception as exc:
        log.warning("Jev unavailable (%s); the word list names the complaint", exc)
        return classify(diagnosis or "", "; ".join(conditions + symptoms))
    choice = max(probabilities, key=probabilities.get)
    sure = probabilities[choice] >= FOLLOWUP_JEV_MIN_PROBABILITY and choice in COMPLAINTS
    return choice if sure else "general"


def open_episode(db: Session, conversation: Conversation, message: Message, level: str,
                 diagnosis: str | None, reviewing_doctor_id=None) -> Episode:
    """Start watching this conversation's illness, or return the episode already watching it.
    The caller commits."""
    existing = active_episode(db, conversation.id)
    if existing is not None:
        return existing
    profile = db.get(Profile, conversation.user_id)
    age = age_on(profile.date_of_birth, date.today()) if profile and profile.date_of_birth else None
    plan = plan_for(_complaint(diagnosis, conversation.scribe_record), age)
    episode = Episode(
        user_id=conversation.user_id, conversation_id=conversation.id, opened_message_id=message.id,
        opened_level=level, reviewing_doctor_id=reviewing_doctor_id, complaint=plan.complaint,
        usual_course=plan.usual_course, return_signs=plan.return_signs,
        checkin_days=list(plan.checkin_days), state="active",
        next_checkin_at=_now() + _wait(days=plan.checkin_days[0]), opened_at=_now(),
    )
    db.add(episode)
    db.flush()
    return episode


def _close(episode: Episode, state: str) -> None:
    episode.state = state
    episode.closed_at = _now()
    episode.next_checkin_at = None
    episode.awaiting_retriage = False


def floor_for_retriage(db: Session, conversation_id, level: str) -> str:
    """A "worse" check-in floors the next turn at level_2. Returns the level to use."""
    episode = active_episode(db, conversation_id)
    if episode is None or not episode.awaiting_retriage:
        return level
    if _last_answer(db, episode).outcome != "retriage":
        return level  # new words at a better/same check-in: the guardian's level stands
    return level if _RANK.get(level, 2) >= 2 else "level_2"


def after_turn(db: Session, conversation: Conversation, message: Message, level: str,
               result: dict) -> Episode | None:
    """Open, keep or close the conversation's episode after a turn. The caller commits."""
    episode = active_episode(db, conversation.id)
    if level != "level_1":
        # A doctor or the emergency path now owns this illness.
        if episode is not None:
            _close(episode, "escalated")
        return None
    if episode is not None and episode.awaiting_retriage:
        # Only new words at a better/same check-in reach here at level_1 ("worse" is floored
        # above it). Triage found them self-care, so the check-in's own answer stands.
        episode.awaiting_retriage = False
        if _last_answer(db, episode).overall == "better":
            _close(episode, "resolved")
            episode = None
    if episode is not None:
        return episode
    clinical = result.get("intent_type") == "clinical_query" and not result.get("is_clarification")
    if not clinical:
        return None
    diagnosis = (result.get("orchestrator_output") or {}).get("probable_diagnosis")
    return open_episode(db, conversation, message, level, diagnosis)


# ── answering a check-in ─────────────────────────────────────────────────────────────

_REPLIES = {
    "resolved": "Good to hear it's getting better. Check-ins for this have stopped. If anything "
                "changes, you can message here any time.",
    "doctor_course": "It hasn't improved in the time this usually takes, so a doctor will review it. "
                     "You'll be told when they have.",
    "doctor_sign": "That sign means a doctor should look at this. A doctor will review it, and if "
                   "you can, arrange to see one today.",
    "urgent": "That is a warning sign that needs urgent care. Go to the nearest emergency department "
              "or call your local emergency number now. Don't wait for this system.",
}
def _schedule_next(episode: Episode, now: datetime) -> None:
    days = episode.checkin_days
    gap = days[episode.checkins_done] - days[episode.checkins_done - 1]
    episode.next_checkin_at = now + _wait(days=gap)


def _own_episode(db: Session, user: User, episode_id) -> Episode:
    try:
        eid = uuid.UUID(str(episode_id))
    except ValueError:
        raise NotFound
    episode = db.get(Episode, eid)
    if episode is None or episode.user_id != user.id:
        raise NotFound
    return episode


def _exchange(db: Session, episode: Episode, said: str, reply: str, level: str) -> Message:
    """Write the check-in into the conversation, so the transcript and later turns see it."""
    db.add(Message(conversation_id=episode.conversation_id, role="user", content=said))
    message = Message(conversation_id=episode.conversation_id, role="assistant", content=reply,
                      triage_level=level)
    db.add(message)
    db.flush()
    return message


def answer_checkin(db: Session, user: User, episode_id, overall: str, signs: list[str],
                   new: str | None) -> dict:
    """Turn a check-in answer into an outcome. Raises NotFound, Conflict or Invalid."""
    episode = _own_episode(db, user, episode_id)
    checkin = open_checkin(db, episode) if episode.state == "active" else None
    if checkin is None:
        raise Conflict("No check-in is due for this right now.")
    by_text = {s["text"]: s for s in episode.return_signs}
    if any(s not in by_text for s in signs):
        raise Invalid("A sign that was not on the list was reported.")

    now = _now()
    new = (new or "").strip() or None
    checkin.answered_at, checkin.overall, checkin.signs_present, checkin.new_text = now, overall, signs, new
    episode.checkins_done += 1
    said = f"Check-in: {OVERALL_WORDS[overall]}."
    if signs:
        said += " Warning signs: " + ", ".join(signs) + "."
    if new:
        said += f" Also: {new}"

    conversation = db.get(Conversation, episode.conversation_id)
    sign_rank = max((_RANK[by_text[s]["level"]] for s in signs), default=0)
    remaining = episode.checkins_done < len(episode.checkin_days)
    message = None

    if sign_rank == 3:
        outcome = "urgent"
        message = _exchange(db, episode, said, _REPLIES["urgent"], "level_3")
        conversation.locked = True
        _close(episode, "escalated")
    elif sign_rank == 2 or (overall == "same" and not remaining):
        outcome = "doctor"
        reply = _REPLIES["doctor_sign"] if sign_rank == 2 else _REPLIES["doctor_course"]
        message = _exchange(db, episode, said, reply, "level_2")
        profile = db.get(Profile, user.id)
        open_case(db, message, user.id, build_background(db, user.id, conversation.id),
                  profile.updated_at if profile else None)
        _close(episode, "escalated")
    elif overall == "worse":
        # Back through the full pipeline, which the patient's next message starts; that
        # turn is floored at level_2 and records the exchange itself.
        outcome = "retriage"
        episode.awaiting_retriage = True
    elif new:
        # Code can't read free text, so new words go through the full pipeline like any
        # message, but not floored: whether they matter is the guardian's call. The patient's
        # next message carries them, and after_turn applies this answer if they prove minor.
        outcome = "reassess"
        episode.awaiting_retriage = True
        if overall == "same":
            _schedule_next(episode, now)
    elif overall == "same":
        outcome = "scheduled"
        _schedule_next(episode, now)
        course = (f" {COMPLAINTS[episode.complaint].label} {episode.usual_course}."
                  if episode.usual_course else "")
        reply = (f"Thanks.{course} We'll check in again on {_day(episode.next_checkin_at)}. If any of the "
                 "warning signs appear before then, don't wait for us: get medical help.")
        message = _exchange(db, episode, said, reply, "level_1")
    else:
        outcome = "resolved"
        message = _exchange(db, episode, said, _REPLIES["resolved"], "level_1")
        _close(episode, "resolved")

    checkin.outcome = outcome
    db.commit()
    return {"outcome": outcome, "message_id": message.id if message else None}


def postpone(db: Session, user: User, episode_id, hours: int) -> None:
    """The patient picks a later time. An open check-in is set aside, not counted."""
    episode = _own_episode(db, user, episode_id)
    if episode.state != "active":
        raise Conflict("This is no longer being followed up.")
    checkin = open_checkin(db, episode)
    if checkin is not None:
        checkin.outcome = "postponed"
    episode.next_checkin_at = _now() + _wait(hours=hours)
    db.commit()


def stop(db: Session, user: User, episode_id) -> None:
    """The patient opts out. Recorded as stopped, never as recovered."""
    episode = _own_episode(db, user, episode_id)
    if episode.state != "active":
        raise Conflict("This is no longer being followed up.")
    checkin = open_checkin(db, episode)
    if checkin is not None:
        checkin.outcome = "stopped"
    _close(episode, "stopped")
    db.commit()


# ── the scheduler ────────────────────────────────────────────────────────────────────

def _link(episode: Episode) -> str:
    return f"{APP_BASE_URL}/chat?c={episode.conversation_id}"


def run_due(now: datetime | None = None) -> int:
    """One pass: open due check-ins, send the one reminder, lapse the silent. Nothing is
    sent during quiet hours unless the demo toggle is on. Returns how many check-ins opened."""
    now = now or _now()
    if not _demo() and _quiet(now):
        return 0
    opened = 0
    with SessionLocal() as db:
        due = db.scalars(select(Episode).where(Episode.state == "active", Episode.next_checkin_at <= now)
                         .with_for_update(skip_locked=True)).all()
        for episode in due:
            db.add(Checkin(episode_id=episode.id, number=episode.checkins_done + 1, opened_at=now))
            episode.next_checkin_at = None
            patient = db.get(User, episode.user_id)
            enqueue(db, patient.email, "checkin_due", name=patient.name, link=_link(episode), reminder=False)
            opened += 1

        waiting = db.execute(
            select(Checkin, Episode).join(Episode, Checkin.episode_id == Episode.id)
            .where(Episode.state == "active", Checkin.outcome.is_(None))
        ).all()
        for checkin, episode in waiting:
            patient = db.get(User, episode.user_id)
            if checkin.reminded_at is None and checkin.opened_at <= now - _wait(hours=FOLLOWUP_REMINDER_HOURS):
                checkin.reminded_at = now
                enqueue(db, patient.email, "checkin_due", name=patient.name, link=_link(episode), reminder=True)
            elif checkin.reminded_at is not None and checkin.reminded_at <= now - _wait(hours=FOLLOWUP_LAPSE_HOURS):
                checkin.outcome = "lapsed"
                _close(episode, "lapsed")
                if episode.reviewing_doctor_id is not None:
                    doctor = db.get(User, episode.reviewing_doctor_id)
                    enqueue(db, doctor.email, "followup_unanswered", name=doctor.name,
                            link=f"{APP_BASE_URL}/doctor")
        db.commit()
    return opened


def start_worker(interval_seconds: float = 5.0) -> threading.Event:
    """Run the scheduler in a background thread. Set the returned event to stop it."""
    stop_event = threading.Event()

    def loop():
        while not stop_event.wait(interval_seconds):
            try:
                run_due()
            except Exception:
                log.exception("follow-up scheduler pass failed")

    threading.Thread(target=loop, name="followup-scheduler", daemon=True).start()
    return stop_event
