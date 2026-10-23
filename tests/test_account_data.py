from sqlalchemy import text

from app.db.models import Conversation
from app.services.patient_context import sync_record
from tests.conftest import PASSWORD, auth, doctor, outbox, patient


def _count(query, **params):
    from app.db.session import engine
    with engine.begin() as conn:
        return conn.execute(text(query), params).scalar_one()


def _uid(client, token):
    return client.get("/me", headers=auth(token)).json()["id"]


def test_export_contains_the_callers_data_and_nobody_elses(client, graph):
    alice = patient(client, email="alice@example.com", name="Alice")
    bob = patient(client, email="bob@example.com", name="Bob")
    client.put("/me/profile", json={"date_of_birth": "1990-02-03", "gender": "female",
                                    "allergies": [{"name": "latex", "detail": None}],
                                    "medications": [], "conditions": []}, headers=auth(alice))
    client.post("/process", json={"text": "alice's question"}, headers=auth(alice))
    client.post("/process", json={"text": "bob's question"}, headers=auth(bob))

    export = client.get("/me/export", headers=auth(alice)).json()

    assert export["account"]["email"] == "alice@example.com"
    assert export["profile"]["allergies"] == [{"name": "latex", "detail": None}]
    [conversation] = export["conversations"]
    assert [m["content"] for m in conversation["messages"]] == ["alice's question", "Rest and drink fluids."]
    assert "bob's question" not in str(export)


def test_deleting_an_account_needs_the_password(client):
    token = patient(client)
    r = client.request("DELETE", "/me", json={"password": "not my password"}, headers=auth(token))
    assert r.status_code == 403
    assert client.get("/me", headers=auth(token)).status_code == 200


def test_deleting_an_account_removes_everything_it_owns_and_nothing_else(client, graph, db):
    doc = doctor(client)
    alice = patient(client, email="alice@example.com", name="Alice")
    bob = patient(client, email="bob@example.com", name="Bob")
    alice_id = _uid(client, alice)
    graph.level = "level_2"
    client.post("/process", json={"text": "alice's question"}, headers=auth(alice))
    client.post("/process", json={"text": "bob's question"}, headers=auth(bob))
    conversation = Conversation(user_id=alice_id)
    db.add(conversation)
    db.commit()
    sync_record(db, alice_id, conversation.id, {"allergies": ["latex"]})
    db.commit()
    for case in client.get("/doctor/cases", headers=auth(doc)).json():
        client.get(f"/doctor/cases/{case['id']}", headers=auth(doc))

    r = client.request("DELETE", "/me", json={"password": PASSWORD}, headers=auth(alice))

    assert r.status_code == 204
    for table in ("profiles", "conversations", "record_items", "sessions", "review_cases"):
        column = "patient_id" if table == "review_cases" else "user_id"
        assert _count(f"SELECT count(*) FROM {table} WHERE {column} = CAST(:u AS uuid)", u=alice_id) == 0, table
    assert _count("SELECT count(*) FROM users WHERE email = 'alice@example.com'") == 0
    assert _count("SELECT count(*) FROM messages WHERE content = 'alice''s question'") == 0
    # Bob is untouched.
    assert _count("SELECT count(*) FROM messages WHERE content = 'bob''s question'") == 1
    assert _count("SELECT count(*) FROM review_cases") == 1
    # The access log keeps that a view happened, but no longer who it was about.
    assert _count("SELECT count(*) FROM access_log") == 2
    assert _count("SELECT count(*) FROM access_log WHERE patient_id IS NULL") == 1
    # The deletion is confirmed to the address that was deleted, and the old session is dead.
    assert len(outbox("account_deleted", to="alice@example.com")) == 1
    assert client.get("/me", headers=auth(alice)).status_code == 401
    assert client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD}).status_code == 401
