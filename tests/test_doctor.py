import json
from datetime import date

from sqlalchemy import text

from app.db.session import engine
from tests.conftest import auth, doctor, outbox, patient


def _sql(query, **params):
    with engine.begin() as conn:
        return conn.execute(text(query), params)


def _level_2_case(client, graph, token, text="I have a cough and fever", conversation_id=None):
    graph.level = "level_2"
    body = {"text": text}
    if conversation_id:
        body["conversation_id"] = conversation_id
    r = client.post("/process", json=body, headers=auth(token))
    assert r.status_code == 200, r.text
    return r.json()["conversation_id"]


def _only_case(client, doc):
    [case] = client.get("/doctor/cases", headers=auth(doc)).json()
    return case


def test_a_doctor_sees_a_pending_case_with_the_patients_question(client, graph):
    doc = doctor(client)
    _level_2_case(client, graph, patient(client))

    case = _only_case(client, doc)

    assert case["status"] == "pending"
    assert case["question"] == "I have a cough and fever"
    assert case["patient_name"] == "Pat"


def test_a_case_shows_the_transcript_up_to_the_case_and_the_background_the_ai_saw(client, graph):
    doc = doctor(client)
    token = patient(client)
    client.put("/me/profile", json={"date_of_birth": date(date.today().year - 61, 1, 1).isoformat(),
                                    "gender": "female", "allergies": [{"name": "penicillin", "detail": None}],
                                    "medications": [], "conditions": []}, headers=auth(token))
    conversation_id = _level_2_case(client, graph, token, text="first question")
    graph.level = "level_1"
    client.post("/process", json={"text": "a later question", "conversation_id": conversation_id},
                headers=auth(token))

    detail = client.get(f"/doctor/cases/{_only_case(client, doc)['id']}", headers=auth(doc)).json()

    assert [m["content"] for m in detail["transcript"]] == ["first question", "Rest and drink fluids."]
    assert "Age: 61" in detail["background"]
    assert "penicillin" in detail["background"]
    assert detail["profile"]["gender"] == "female"
    assert detail["profile_changed_since"] is False


def test_a_case_says_when_the_profile_changed_after_the_ai_answered(client, graph):
    doc = doctor(client)
    token = patient(client)
    _level_2_case(client, graph, token)
    _sql("UPDATE profiles SET updated_at = now() + interval '1 minute'")

    detail = client.get(f"/doctor/cases/{_only_case(client, doc)['id']}", headers=auth(doc)).json()

    assert detail["profile_changed_since"] is True


def test_opening_a_case_is_logged(client, graph):
    doc = doctor(client)
    _level_2_case(client, graph, patient(client))
    case_id = _only_case(client, doc)["id"]

    client.get(f"/doctor/cases/{case_id}", headers=auth(doc))

    [row] = _sql("SELECT action, case_id::text AS case_id, patient_id IS NOT NULL AS has_patient "
                 "FROM access_log").mappings().all()
    assert dict(row) == {"action": "view_case", "case_id": case_id, "has_patient": True}


def test_a_patient_cannot_use_the_doctor_endpoints(client, graph):
    doctor(client)
    token = patient(client)
    _level_2_case(client, graph, token)

    assert client.get("/doctor/cases", headers=auth(token)).status_code == 403


def test_a_doctor_cannot_open_a_case_another_doctor_already_reviewed(client, graph):
    doc_a = doctor(client, email="a@example.com")
    doc_b = doctor(client, email="b@example.com")
    _level_2_case(client, graph, patient(client))
    case_id = _only_case(client, doc_a)["id"]
    client.patch(f"/doctor/cases/{case_id}", json={"status": "verified", "notes": "ok"}, headers=auth(doc_a))

    assert client.get(f"/doctor/cases/{case_id}", headers=auth(doc_b)).status_code == 404
    assert client.get("/doctor/cases", headers=auth(doc_b)).json() == []
    assert client.get(f"/doctor/cases/{case_id}", headers=auth(doc_a)).status_code == 200
    assert [c["id"] for c in client.get("/doctor/cases", headers=auth(doc_a)).json()] == [case_id]


def test_approving_a_case_reaches_the_patient_and_the_email_carries_no_health_details(client, graph):
    doc = doctor(client)
    token = patient(client)
    graph.response = "This sounds like community-acquired pneumonia."
    conversation_id = _level_2_case(client, graph, token)
    case_id = _only_case(client, doc)["id"]

    r = client.patch(f"/doctor/cases/{case_id}", json={"status": "verified", "notes": "Agree; see a GP today."},
                     headers=auth(doc))

    assert r.status_code == 200
    review = client.get(f"/conversations/{conversation_id}", headers=auth(token)).json()["messages"][1]["review"]
    assert review["status"] == "verified"
    assert review["notes"] == "Agree; see a GP today."
    [email] = outbox("case_reviewed", to="pat@example.com")
    flat = json.dumps(email["payload"]).lower()
    for health_detail in ("pneumonia", "cough", "fever", "gp today", "verified", "agree"):
        assert health_detail not in flat


def test_a_reviewed_case_cannot_be_reviewed_again(client, graph):
    doc = doctor(client)
    _level_2_case(client, graph, patient(client))
    case_id = _only_case(client, doc)["id"]
    client.patch(f"/doctor/cases/{case_id}", json={"status": "rejected", "notes": "needs exam"}, headers=auth(doc))

    r = client.patch(f"/doctor/cases/{case_id}", json={"status": "verified", "notes": "changed my mind"},
                     headers=auth(doc))

    assert r.status_code == 409
    assert len(outbox("case_reviewed")) == 1


def test_a_case_can_only_be_marked_verified_or_rejected(client, graph):
    doc = doctor(client)
    _level_2_case(client, graph, patient(client))
    case_id = _only_case(client, doc)["id"]

    assert client.patch(f"/doctor/cases/{case_id}", json={"status": "pending"}, headers=auth(doc)).status_code == 422


def test_the_public_status_reports_only_the_waiting_count(client, graph):
    doctor(client)
    token = patient(client)
    _level_2_case(client, graph, token)
    _level_2_case(client, graph, token)

    assert client.get("/status").json() == {"waiting": 2}
