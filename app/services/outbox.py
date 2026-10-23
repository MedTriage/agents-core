"""Email outbox: queue in the caller's transaction, send from a background loop."""
from sqlalchemy.orm import Session

from app.db.models import EmailOutbox


def enqueue(db: Session, recipient: str, template: str, **payload) -> None:
    """Queue an email. Commits with the caller's transaction, so an email exists exactly
    when the event that caused it does. The payload is rendering input — names and links
    — and must never carry health information."""
    db.add(EmailOutbox(recipient=recipient, template=template, payload=payload))
