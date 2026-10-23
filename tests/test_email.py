import smtplib
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from app.db.session import engine
from app.services import mailer
from tests.conftest import outbox, signup

LINKED = ["verify_email", "password_reset", "doctor_invite", "case_reviewed", "cases_waiting"]
UNLINKED = ["password_changed", "account_deleted"]
LINK = "http://localhost:3000/x?token=abc123"


class RecordingTransport:
    def __init__(self, fail_for=()):
        self.sent, self.fail_for = [], set(fail_for)

    def send(self, message):
        if message["To"] in self.fail_for:
            raise smtplib.SMTPServerDisconnected("connection dropped")
        self.sent.append(message)


def _parts(message):
    html = message.get_body(("html",)).get_content()
    plain = message.get_body(("plain",)).get_content()
    images = [p for p in message.walk() if p.get_content_type() == "image/png"]
    return html, plain, images


# ── rendering ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("template", LINKED)
def test_a_linked_email_carries_its_link_in_both_versions_and_the_logo_inline(template):
    message = mailer.build_message("pat@example.com", template, {"name": "Pat", "link": LINK})

    html, plain, images = _parts(message)
    assert message["To"] == "pat@example.com"
    assert message["Subject"]
    assert "Pat" in html and "Pat" in plain
    assert LINK.replace("&", "&amp;") in html
    assert LINK in plain
    [logo] = images
    assert f"cid:{logo['Content-ID'].strip('<>')}" in html


@pytest.mark.parametrize("template", UNLINKED)
def test_an_unlinked_email_renders_with_the_logo(template):
    html, plain, images = _parts(mailer.build_message("pat@example.com", template, {"name": "Pat"}))
    assert "Pat" in html and "Pat" in plain
    assert len(images) == 1


def test_each_email_has_its_own_subject():
    subjects = {mailer.build_message("a@b.co", t, {"name": "P", "link": LINK})["Subject"]
                for t in LINKED + UNLINKED}
    assert len(subjects) == len(LINKED + UNLINKED)


def test_a_name_cannot_inject_html_into_an_email():
    html, _, _ = _parts(mailer.build_message("pat@example.com", "verify_email",
                                             {"name": "<script>x()</script>", "link": LINK}))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ── sending ──────────────────────────────────────────────────────────────────────────

def test_a_queued_email_is_sent_once_and_marked_sent(client):
    signup(client)
    transport = RecordingTransport()

    assert mailer.process_outbox(transport) == 1
    assert mailer.process_outbox(transport) == 0

    [message] = transport.sent
    assert message["To"] == "pat@example.com"
    assert outbox("verify_email")[0]["status"] == "sent"


def test_a_failed_send_is_retried_later_and_given_up_after_five_attempts(client):
    signup(client)
    transport = RecordingTransport(fail_for={"pat@example.com"})

    assert mailer.process_outbox(transport) == 0
    [row] = outbox()
    assert (row["status"], row["attempts"]) == ("pending", 1)

    # Not due again yet: a retry waits for its backoff.
    mailer.process_outbox(transport)
    assert outbox()[0]["attempts"] == 1

    for _ in range(4):
        with engine.begin() as conn:
            conn.execute(text("UPDATE email_outbox SET next_attempt_at = now() - interval '1 second'"))
        mailer.process_outbox(transport)

    [row] = outbox()
    assert (row["status"], row["attempts"]) == ("failed", 5)


def test_one_failing_address_does_not_hold_up_the_others(client):
    signup(client, email="bad@example.com")
    signup(client, email="good@example.com")

    sent = mailer.process_outbox(RecordingTransport(fail_for={"bad@example.com"}))

    assert sent == 1
    assert outbox(to="good@example.com")[0]["status"] == "sent"
    assert outbox(to="bad@example.com")[0]["status"] == "pending"


def test_without_smtp_credentials_nothing_is_sent_and_nothing_fails(client, monkeypatch):
    monkeypatch.setattr(mailer, "SMTP_USER", None)
    monkeypatch.setattr(mailer, "SMTP_APP_PASSWORD", None)
    signup(client)

    assert mailer.transport_from_config() is None
    assert mailer.process_outbox(None) == 0
    assert outbox()[0]["status"] == "pending"
