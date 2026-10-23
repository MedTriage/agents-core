from datetime import date, datetime, timedelta, timezone

from app.db.models import Conversation, RecordItem
from app.services.patient_context import build_background, sync_record
from tests.conftest import auth, doctor, patient


def _uid(client, token):
    return client.get("/me", headers=auth(token)).json()["id"]


def _conversation(db, user_id):
    c = Conversation(user_id=user_id)
    db.add(c)
    db.commit()
    return c


def _scribe(**fields):
    record = {"current_conditions": [], "past_conditions": [], "symptoms": [], "medications": [],
              "allergies": [], "demographics": {}, "notes": [], "turn": 1}
    record.update(fields)
    return record


FULL_PROFILE = {
    "date_of_birth": None,  # set per test
    "gender": "female",
    "allergies": [{"name": "Penicillin", "detail": "rash"}],
    "medications": [{"name": "Metformin", "detail": None}],
    "conditions": [{"name": "Type 2 diabetes", "detail": None}],
}


# ── profile ──────────────────────────────────────────────────────────────────────────

def test_a_new_patient_has_an_empty_profile(client):
    token = patient(client)
    assert client.get("/me/profile", headers=auth(token)).json() == {
        "date_of_birth": None, "gender": None, "allergies": [], "medications": [], "conditions": [],
    }


def test_a_patient_can_save_and_read_back_their_profile(client):
    token = patient(client)
    body = {**FULL_PROFILE, "date_of_birth": "1984-05-17"}
    assert client.put("/me/profile", json=body, headers=auth(token)).status_code == 200
    assert client.get("/me/profile", headers=auth(token)).json() == body


def test_profile_rejects_an_unknown_gender(client):
    token = patient(client)
    r = client.put("/me/profile", json={**FULL_PROFILE, "gender": "robot"}, headers=auth(token))
    assert r.status_code == 422


def test_profile_rejects_a_date_of_birth_in_the_future(client):
    token = patient(client)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    r = client.put("/me/profile", json={**FULL_PROFILE, "date_of_birth": tomorrow}, headers=auth(token))
    assert r.status_code == 422


def test_a_doctor_has_no_patient_profile(client):
    token = doctor(client)
    assert client.get("/me/profile", headers=auth(token)).status_code == 403


# ── the background block the agents see ──────────────────────────────────────────────

def test_background_carries_age_and_gender_from_the_profile(client, db):
    token = patient(client)
    born = date(date.today().year - 40, 1, 1).isoformat()
    client.put("/me/profile", json={**FULL_PROFILE, "date_of_birth": born}, headers=auth(token))
    uid = _uid(client, token)

    background = build_background(db, uid, _conversation(db, uid).id)

    assert "Age: 40" in background
    assert "Gender: female" in background
    assert "Penicillin (rash)" in background
    assert "Metformin" in background
    assert "Type 2 diabetes" in background


def test_background_says_when_age_was_not_given(client, db):
    token = patient(client)
    client.put("/me/profile", json={"gender": "non_binary", "allergies": [], "medications": [],
                                    "conditions": [], "date_of_birth": None}, headers=auth(token))
    uid = _uid(client, token)

    background = build_background(db, uid, _conversation(db, uid).id)

    assert "Age: not provided" in background
    assert "Gender: non-binary" in background


def test_background_is_absent_when_nothing_is_known(client, db):
    token = patient(client)
    uid = _uid(client, token)
    assert build_background(db, uid, _conversation(db, uid).id) is None


def test_background_lists_an_allergy_from_an_earlier_chat_as_unconfirmed(client, db):
    token = patient(client)
    uid = _uid(client, token)
    earlier, today = _conversation(db, uid), _conversation(db, uid)
    sync_record(db, uid, earlier.id, _scribe(allergies=["ibuprofen"]))
    db.commit()

    background = build_background(db, uid, today.id)

    stated, _, unconfirmed = background.partition("not confirmed by the patient")
    assert "ibuprofen" in unconfirmed
    assert "ibuprofen" not in stated


def test_background_leaves_out_dismissed_items_and_this_conversations_own_items(client, db):
    token = patient(client)
    uid = _uid(client, token)
    earlier, today = _conversation(db, uid), _conversation(db, uid)
    sync_record(db, uid, earlier.id, _scribe(allergies=["ibuprofen"], medications=["aspirin"]))
    sync_record(db, uid, today.id, _scribe(current_conditions=["asthma"]))
    db.commit()
    aspirin = db.query(RecordItem).filter_by(norm_text="aspirin").one()
    client.post(f"/me/suggestions/{aspirin.id}/dismiss", headers=auth(token))

    background = build_background(db, uid, today.id)

    assert "ibuprofen" in background
    assert "aspirin" not in background
    assert "asthma" not in background  # already in this conversation's own scribe record


# ── the record and its suggestions ───────────────────────────────────────────────────

def test_record_keeps_conditions_medications_and_allergies_but_not_symptoms(client, db):
    token = patient(client)
    uid = _uid(client, token)
    conv = _conversation(db, uid)

    sync_record(db, uid, conv.id, _scribe(
        current_conditions=["asthma"], past_conditions=["appendicitis"],
        medications=["salbutamol inhaler"], allergies=["peanuts"],
        symptoms=[{"text": "wheeze", "first_turn": 1, "last_turn": 1}],
    ))
    db.commit()

    items = {(i.kind, i.text) for i in db.query(RecordItem).filter_by(user_id=uid)}
    assert items == {("condition", "asthma"), ("condition", "appendicitis"),
                     ("medication", "salbutamol inhaler"), ("allergy", "peanuts")}


def test_something_new_is_suggested_and_something_already_in_the_profile_is_not(client, db):
    token = patient(client)
    client.put("/me/profile", json={**FULL_PROFILE, "date_of_birth": "1984-05-17"}, headers=auth(token))
    uid = _uid(client, token)

    sync_record(db, uid, _conversation(db, uid).id,
                _scribe(medications=["metformin 500mg twice daily", "lisinopril"]))
    db.commit()

    suggested = [s["text"] for s in client.get("/me/suggestions", headers=auth(token)).json()]
    assert suggested == ["lisinopril"]


def test_the_record_never_changes_the_profile(client, db):
    token = patient(client)
    before = client.get("/me/profile", headers=auth(token)).json()
    uid = _uid(client, token)

    sync_record(db, uid, _conversation(db, uid).id,
                _scribe(allergies=["latex"], medications=["warfarin"], current_conditions=["gout"]))
    db.commit()

    assert client.get("/me/profile", headers=auth(token)).json() == before


def test_accepting_a_suggestion_adds_it_to_the_right_profile_list(client, db):
    token = patient(client)
    uid = _uid(client, token)
    sync_record(db, uid, _conversation(db, uid).id, _scribe(allergies=["latex"]))
    db.commit()
    [suggestion] = client.get("/me/suggestions", headers=auth(token)).json()

    assert client.post(f"/me/suggestions/{suggestion['id']}/accept", headers=auth(token)).status_code == 200

    profile = client.get("/me/profile", headers=auth(token)).json()
    assert profile["allergies"] == [{"name": "latex", "detail": None}]
    assert profile["medications"] == [] and profile["conditions"] == []
    assert client.get("/me/suggestions", headers=auth(token)).json() == []


def test_a_dismissed_suggestion_is_not_suggested_again_when_mentioned_again(client, db):
    token = patient(client)
    uid = _uid(client, token)
    sync_record(db, uid, _conversation(db, uid).id, _scribe(medications=["vitamin d"]))
    db.commit()
    [suggestion] = client.get("/me/suggestions", headers=auth(token)).json()
    client.post(f"/me/suggestions/{suggestion['id']}/dismiss", headers=auth(token))

    sync_record(db, uid, _conversation(db, uid).id, _scribe(medications=["Vitamin D"]))
    db.commit()

    assert client.get("/me/suggestions", headers=auth(token)).json() == []


def test_a_patient_cannot_act_on_another_patients_suggestion(client, db):
    alice = patient(client, email="alice@example.com")
    bob = patient(client, email="bob@example.com")
    alice_id = _uid(client, alice)
    sync_record(db, alice_id, _conversation(db, alice_id).id, _scribe(allergies=["latex"]))
    db.commit()
    [suggestion] = client.get("/me/suggestions", headers=auth(alice)).json()

    assert client.post(f"/me/suggestions/{suggestion['id']}/accept", headers=auth(bob)).status_code == 404
    assert client.get("/me/profile", headers=auth(bob)).json()["allergies"] == []
    assert client.get("/me/profile", headers=auth(alice)).json()["allergies"] == []


def test_mentioning_an_item_again_updates_when_it_was_last_mentioned(client, db):
    token = patient(client)
    uid = _uid(client, token)
    sync_record(db, uid, _conversation(db, uid).id, _scribe(allergies=["latex"]))
    db.commit()
    item = db.query(RecordItem).filter_by(user_id=uid).one()
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    item.last_mentioned_at = long_ago
    db.commit()

    sync_record(db, uid, _conversation(db, uid).id, _scribe(allergies=["Latex"]))
    db.commit()
    db.refresh(item)

    assert item.last_mentioned_at > long_ago
    assert db.query(RecordItem).filter_by(user_id=uid).count() == 1
