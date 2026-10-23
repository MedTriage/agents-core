"""Sign-up, verification, login, sessions, password reset and doctor invitations."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import (
    APP_BASE_URL, INVITE_TOKEN_DAYS, LOGIN_FAILURE_WINDOW_MINUTES, LOGIN_MAX_FAILURES,
    LOGIN_MAX_FAILURES_PER_IP, RESET_TOKEN_HOURS, SESSION_IDLE_DAYS, SESSION_MAX_DAYS,
    VERIFY_TOKEN_HOURS,
)
from app.db.models import EmailToken, LoginAttempt, Profile, User, UserSession
from app.services.outbox import enqueue
from app.services.security import hash_password, hash_token, new_token, verify_password


class EmailTaken(Exception):
    pass


class InvalidCredentials(Exception):
    pass


class RateLimited(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _link(path: str, token: str) -> str:
    return f"{APP_BASE_URL}{path}?token={token}"


def _issue_token(db: Session, user: User, purpose: str, ttl: timedelta) -> str:
    raw = new_token()
    db.add(EmailToken(user_id=user.id, purpose=purpose, token_hash=hash_token(raw),
                      expires_at=_now() + ttl))
    return raw


def _consume_token(db: Session, raw: str, purpose: str) -> User | None:
    """Spend a single-use emailed token. None if it is unknown, of another purpose,
    already used or expired — the caller cannot tell which, and neither can an attacker."""
    row = db.scalar(
        select(EmailToken)
        .where(EmailToken.token_hash == hash_token(raw), EmailToken.purpose == purpose)
        .with_for_update()
    )
    if row is None or row.used_at is not None or row.expires_at <= _now():
        return None
    row.used_at = _now()
    return db.get(User, row.user_id)


# ── sign-up and verification ─────────────────────────────────────────────────────────

def _send_verification(db: Session, user: User) -> None:
    token = _issue_token(db, user, "verify", timedelta(hours=VERIFY_TOKEN_HOURS))
    enqueue(db, user.email, "verify_email", name=user.name, link=_link("/verify-email", token))


def signup(db: Session, email: str, password: str, name: str) -> User:
    """Create a patient account. Public sign-up never creates a doctor."""
    email = normalize_email(email)
    if db.scalar(select(User.id).where(User.email == email)):
        raise EmailTaken
    user = User(email=email, password_hash=hash_password(password), role="patient",
                name=name.strip())
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise EmailTaken
    db.add(Profile(user_id=user.id))
    _send_verification(db, user)
    db.commit()
    return user


def verify_email(db: Session, raw: str) -> bool:
    user = _consume_token(db, raw, "verify")
    if user is None:
        db.rollback()
        return False
    user.email_verified_at = user.email_verified_at or _now()
    db.commit()
    return True


def resend_verification(db: Session, email: str) -> None:
    user = db.scalar(select(User).where(User.email == normalize_email(email)))
    if user is not None and user.email_verified_at is None and user.role == "patient":
        _send_verification(db, user)
        db.commit()


# ── login and sessions ───────────────────────────────────────────────────────────────

def _recent_failures(db: Session, column, value) -> int:
    since = _now() - timedelta(minutes=LOGIN_FAILURE_WINDOW_MINUTES)
    return db.scalar(
        select(func.count()).select_from(LoginAttempt)
        .where(column == value, LoginAttempt.success.is_(False), LoginAttempt.at > since)
    )


def login(db: Session, email: str, password: str, ip: str) -> tuple[str, User]:
    email = normalize_email(email)
    if (_recent_failures(db, LoginAttempt.email, email) >= LOGIN_MAX_FAILURES
            or _recent_failures(db, LoginAttempt.ip, ip) >= LOGIN_MAX_FAILURES_PER_IP):
        raise RateLimited

    user = db.scalar(select(User).where(User.email == email))
    ok = verify_password(user.password_hash if user else None, password)
    db.add(LoginAttempt(email=email, ip=ip, success=ok))
    if not ok:
        db.commit()
        raise InvalidCredentials

    # A success clears this account's failure count; the per-IP count is left alone.
    db.execute(delete(LoginAttempt).where(LoginAttempt.email == email,
                                          LoginAttempt.success.is_(False)))
    raw = new_token()
    db.add(UserSession(user_id=user.id, token_hash=hash_token(raw),
                       expires_at=_now() + timedelta(days=SESSION_IDLE_DAYS)))
    db.commit()
    return raw, user


def authenticate(db: Session, raw: str) -> tuple[User, UserSession] | None:
    session = db.scalar(select(UserSession).where(UserSession.token_hash == hash_token(raw)))
    now = _now()
    if session is None:
        return None
    hard_cap = session.created_at + timedelta(days=SESSION_MAX_DAYS)
    if session.expires_at <= now or hard_cap <= now:
        return None

    # Slide the idle expiry forward, at most every few minutes so a busy page does not
    # write on every request.
    if now - session.last_seen_at > timedelta(minutes=5):
        session.last_seen_at = now
        session.expires_at = min(now + timedelta(days=SESSION_IDLE_DAYS), hard_cap)
        db.commit()
    return db.get(User, session.user_id), session


def logout(db: Session, session: UserSession) -> None:
    db.delete(session)
    db.commit()


# ── password reset ───────────────────────────────────────────────────────────────────

def request_password_reset(db: Session, email: str) -> None:
    user = db.scalar(select(User).where(User.email == normalize_email(email)))
    if user is not None and user.password_hash is not None:
        token = _issue_token(db, user, "reset", timedelta(hours=RESET_TOKEN_HOURS))
        enqueue(db, user.email, "password_reset", name=user.name,
                link=_link("/reset-password", token))
        db.commit()


def _set_password(db: Session, user: User, password: str) -> None:
    """Every existing session ends: whoever knew the old password is logged out."""
    user.password_hash = hash_password(password)
    db.execute(delete(UserSession).where(UserSession.user_id == user.id))
    enqueue(db, user.email, "password_changed", name=user.name)


def reset_password(db: Session, raw: str, password: str) -> bool:
    user = _consume_token(db, raw, "reset")
    if user is None:
        db.rollback()
        return False
    _set_password(db, user, password)
    db.commit()
    return True


# ── doctors ──────────────────────────────────────────────────────────────────────────

def invite_doctor(db: Session, email: str, name: str) -> User:
    """Create (or re-invite) a doctor account. Only reachable from the admin CLI."""
    email = normalize_email(email)
    user = db.scalar(select(User).where(User.email == email))
    if user is not None and user.role != "doctor":
        raise EmailTaken
    if user is None:
        user = User(email=email, role="doctor", name=name.strip())
        db.add(user)
        db.flush()
    token = _issue_token(db, user, "invite", timedelta(days=INVITE_TOKEN_DAYS))
    enqueue(db, user.email, "doctor_invite", name=user.name, link=_link("/invite", token))
    db.commit()
    return user


def accept_invite(db: Session, raw: str, password: str) -> bool:
    user = _consume_token(db, raw, "invite")
    if user is None:
        db.rollback()
        return False
    user.password_hash = hash_password(password)
    user.email_verified_at = user.email_verified_at or _now()
    db.commit()
    return True
