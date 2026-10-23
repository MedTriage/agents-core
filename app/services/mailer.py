"""Rendering and sending the outbox.

Every email is HTML with a plain-text alternative, and the logo travels inside the message
as an inline attachment: the system runs on localhost, and Gmail cannot fetch an image
from someone else's localhost.
"""
import logging
import smtplib
import ssl
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from sqlalchemy import func, select

from app.config import (
    EMAIL_FROM_NAME, EMAIL_MAX_ATTEMPTS, SMTP_APP_PASSWORD, SMTP_HOST, SMTP_PORT, SMTP_USER,
)
from app.db.models import EmailOutbox
from app.db.session import SessionLocal

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "email"
_LOGO = (TEMPLATE_DIR / "logo.png").read_bytes()
_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html"]),
    undefined=StrictUndefined,
)

SUBJECTS = {
    "verify_email": "Confirm your email for MedTriage",
    "password_reset": "Reset your MedTriage password",
    "password_changed": "Your MedTriage password was changed",
    "doctor_invite": "You've been invited to review cases on MedTriage",
    "case_reviewed": "A doctor has reviewed your question",
    "cases_waiting": "Cases are waiting for review on MedTriage",
    "account_deleted": "Your MedTriage account has been deleted",
}


def build_message(recipient: str, template: str, payload: dict) -> EmailMessage:
    cid = make_msgid(domain="medtriage.local")
    html = _env.get_template(f"{template}.html").render(**payload, logo_cid=cid[1:-1])
    text = _env.get_template(f"{template}.txt").render(**payload)

    message = EmailMessage()
    message["Subject"] = SUBJECTS[template]
    message["From"] = formataddr((EMAIL_FROM_NAME, SMTP_USER or "medtriage@localhost"))
    message["To"] = recipient
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    message.get_payload()[1].add_related(_LOGO, "image", "png", cid=cid)
    return message


class SmtpTransport:
    """Gmail SMTP with STARTTLS and an app password."""

    def __init__(self, host: str, port: int, user: str, password: str):
        self.host, self.port, self.user, self.password = host, port, user, password

    def send(self, message: EmailMessage) -> None:
        with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(self.user, self.password)
            smtp.send_message(message)


def transport_from_config() -> SmtpTransport | None:
    if not (SMTP_USER and SMTP_APP_PASSWORD):
        return None
    return SmtpTransport(SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_APP_PASSWORD)


def process_outbox(transport, batch: int = 20) -> int:
    """Send what is due. A failure backs off (1, 2, 4, 8 minutes) and is given up after
    EMAIL_MAX_ATTEMPTS; one bad address never holds up the rest. Returns how many sent."""
    if transport is None:
        return 0
    sent = 0
    with SessionLocal() as db:
        due = db.scalars(
            select(EmailOutbox)
            .where(EmailOutbox.status == "pending", EmailOutbox.next_attempt_at <= func.now())
            .order_by(EmailOutbox.id).limit(batch)
            .with_for_update(skip_locked=True)
        ).all()
        for row in due:
            now = datetime.now(timezone.utc)
            row.attempts += 1
            try:
                transport.send(build_message(row.recipient, row.template, row.payload))
            except Exception as error:
                row.last_error = f"{type(error).__name__}: {error}"[:500]
                if row.attempts >= EMAIL_MAX_ATTEMPTS:
                    row.status = "failed"
                else:
                    row.next_attempt_at = now + timedelta(minutes=2 ** (row.attempts - 1))
            else:
                row.status, row.sent_at, row.last_error = "sent", now, None
                sent += 1
        db.commit()
    return sent


def start_worker(interval_seconds: float = 5.0) -> threading.Event:
    """Run the outbox in a background thread. Set the returned event to stop it."""
    stop = threading.Event()
    transport = transport_from_config()
    if transport is None:
        log.warning("SMTP_USER / SMTP_APP_PASSWORD not set: emails will queue but not send")

    def loop():
        while not stop.wait(interval_seconds):
            try:
                process_outbox(transport)
            except Exception:
                log.exception("email outbox pass failed")

    threading.Thread(target=loop, name="email-outbox", daemon=True).start()
    return stop
