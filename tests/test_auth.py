from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.db.session import engine
from tests.conftest import PASSWORD, auth, doctor, link_token, outbox, patient, signup


def _login(client, email="pat@example.com", password=PASSWORD):
    return client.post("/auth/login", json={"email": email, "password": password})


# ── sign-up ──────────────────────────────────────────────────────────────────────────

def test_signup_creates_unverified_account_and_queues_one_verification_email(client):
    assert signup(client).status_code == 201

    me = client.get("/me", headers=auth(_login(client).json()["token"])).json()
    assert me["email"] == "pat@example.com"
    assert me["role"] == "patient"
    assert me["verified"] is False
    emails = outbox(to="pat@example.com")
    assert [e["template"] for e in emails] == ["verify_email"]
    assert emails[0]["payload"]["link"].startswith("http://localhost:3000/verify-email?token=")


def test_signup_rejects_an_email_already_registered_in_any_case(client):
    assert signup(client, email="pat@example.com").status_code == 201
    assert signup(client, email="PAT@Example.com").status_code == 409


def test_signup_requires_consent(client):
    r = client.post("/auth/signup", json={"email": "pat@example.com", "password": PASSWORD,
                                          "name": "Pat", "consent": False})
    assert r.status_code == 422


def test_signup_rejects_a_password_under_ten_characters(client):
    assert signup(client, password="short12").status_code == 422


def test_password_is_not_stored_in_plain_text(client):
    signup(client)
    with engine.begin() as conn:
        stored = conn.execute(text("SELECT password_hash FROM users")).scalar_one()
    assert PASSWORD not in stored
    assert stored.startswith("$argon2id$")


# ── verification ─────────────────────────────────────────────────────────────────────

def test_verification_link_verifies_once_and_only_once(client):
    signup(client)
    token = link_token(outbox("verify_email")[-1])

    assert client.post("/auth/verify", json={"token": token}).status_code == 200
    assert client.post("/auth/verify", json={"token": token}).status_code == 400

    me = client.get("/me", headers=auth(_login(client).json()["token"])).json()
    assert me["verified"] is True


def test_expired_verification_link_is_rejected(client):
    signup(client)
    token = link_token(outbox("verify_email")[-1])
    with engine.begin() as conn:
        conn.execute(text("UPDATE email_tokens SET expires_at = now() - interval '1 minute'"))

    assert client.post("/auth/verify", json={"token": token}).status_code == 400


def test_resend_verification_answers_the_same_for_unknown_and_known_emails(client):
    signup(client)
    known = client.post("/auth/verify/resend", json={"email": "pat@example.com"})
    unknown = client.post("/auth/verify/resend", json={"email": "nobody@example.com"})

    assert (known.status_code, known.json()) == (unknown.status_code, unknown.json())
    assert len(outbox("verify_email", to="pat@example.com")) == 2
    assert outbox(to="nobody@example.com") == []


# ── login and sessions ───────────────────────────────────────────────────────────────

def test_login_with_a_wrong_password_is_rejected(client):
    signup(client)
    assert _login(client, password="wrong password!").status_code == 401


def test_login_for_an_unknown_email_is_rejected_the_same_way(client):
    wrong = _login(client, email="nobody@example.com")
    assert wrong.status_code == 401


def test_login_is_refused_after_five_failures_even_with_the_right_password(client):
    signup(client)
    for _ in range(5):
        assert _login(client, password="wrong password!").status_code == 401

    assert _login(client).status_code == 429


def test_session_token_is_stored_only_as_a_hash(client):
    signup(client)
    token = _login(client).json()["token"]
    with engine.begin() as conn:
        stored = conn.execute(text("SELECT token_hash FROM sessions")).scalar_one()
    assert token not in stored


def test_logout_revokes_the_session(client):
    token = patient(client)
    assert client.get("/me", headers=auth(token)).status_code == 200

    assert client.post("/auth/logout", headers=auth(token)).status_code == 204
    assert client.get("/me", headers=auth(token)).status_code == 401


def test_an_expired_session_is_rejected(client):
    token = patient(client)
    with engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET expires_at = now() - interval '1 minute'"))

    assert client.get("/me", headers=auth(token)).status_code == 401


def test_a_session_never_outlives_its_absolute_cap(client):
    token = patient(client)
    old = datetime.now(timezone.utc) - timedelta(days=31)
    with engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET created_at = :old"), {"old": old})

    assert client.get("/me", headers=auth(token)).status_code == 401


def test_requests_without_a_token_are_rejected(client):
    assert client.get("/me").status_code == 401
    assert client.get("/me", headers=auth("not-a-real-token")).status_code == 401


# ── password reset ───────────────────────────────────────────────────────────────────

def test_forgot_password_answers_the_same_whether_or_not_the_account_exists(client):
    signup(client)
    known = client.post("/auth/password/forgot", json={"email": "pat@example.com"})
    unknown = client.post("/auth/password/forgot", json={"email": "nobody@example.com"})

    assert (known.status_code, known.json()) == (unknown.status_code, unknown.json())
    assert len(outbox("password_reset", to="pat@example.com")) == 1
    assert outbox(to="nobody@example.com") == []


def test_password_reset_revokes_sessions_and_sends_a_change_notice(client):
    old_session = patient(client)
    client.post("/auth/password/forgot", json={"email": "pat@example.com"})
    token = link_token(outbox("password_reset")[-1])

    r = client.post("/auth/password/reset", json={"token": token, "password": "a brand new password"})
    assert r.status_code == 200

    assert client.get("/me", headers=auth(old_session)).status_code == 401
    assert _login(client).status_code == 401
    assert _login(client, password="a brand new password").status_code == 200
    assert len(outbox("password_changed", to="pat@example.com")) == 1


def test_a_reset_link_cannot_be_used_twice(client):
    patient(client)
    client.post("/auth/password/forgot", json={"email": "pat@example.com"})
    token = link_token(outbox("password_reset")[-1])

    assert client.post("/auth/password/reset", json={"token": token, "password": "a brand new password"}).status_code == 200
    assert client.post("/auth/password/reset", json={"token": token, "password": "another password!"}).status_code == 400


def test_a_verification_token_cannot_be_used_to_reset_a_password(client):
    signup(client)
    verify_token = link_token(outbox("verify_email")[-1])

    r = client.post("/auth/password/reset", json={"token": verify_token, "password": "a brand new password"})
    assert r.status_code == 400


# ── doctors ──────────────────────────────────────────────────────────────────────────

def test_an_invited_doctor_sets_a_password_and_logs_in_as_a_doctor(client):
    token = doctor(client)
    me = client.get("/me", headers=auth(token)).json()
    assert me["role"] == "doctor"
    assert me["verified"] is True


def test_there_is_no_public_way_to_sign_up_as_a_doctor(client):
    r = client.post("/auth/signup", json={"email": "doc@example.com", "password": PASSWORD,
                                          "name": "Dr X", "consent": True, "role": "doctor"})
    assert r.status_code == 201
    token = _login(client, email="doc@example.com").json()["token"]
    assert client.get("/me", headers=auth(token)).json()["role"] == "patient"


def test_an_invite_link_cannot_be_accepted_twice(client):
    doctor(client)
    token = link_token(outbox("doctor_invite")[-1])
    r = client.post("/auth/invite/accept", json={"token": token, "password": "taking it over now"})
    assert r.status_code == 400
