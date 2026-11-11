"""The patient dashboard's data: health readings the patient enters, and the follow-up calendar.
The latest readings also reach the agents, through the patient background."""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.services import followup
from tests.conftest import PASSWORD, auth, doctor, patient

IST = ZoneInfo("Asia/Kolkata")


def _at(days: int, hour: int = 10) -> datetime:
    local = (datetime.now(IST) + timedelta(days=days)).replace(hour=hour, minute=0, second=0, microsecond=0)
    return local.astimezone(timezone.utc)


def _add(client, token, **body):
    return client.post("/me/readings", json=body, headers=auth(token))


def _weight(client, token, value=72, on=None):
    r = _add(client, token, kind="weight", value=value, **({"measured_on": on} if on else {}))
    assert r.status_code == 201, r.text
    return r.json()


# ── readings ─────────────────────────────────────────────────────────────────────────

def test_a_patient_starts_with_no_readings(client):
    assert client.get("/me/readings", headers=auth(patient(client))).json() == []


def test_a_reading_is_stored_with_its_unit_and_todays_date_by_default(client):
    token = patient(client)

    reading = _weight(client, token, 72.5)

    assert reading["kind"] == "weight"
    assert reading["value"] == 72.5
    assert reading["value2"] is None
    assert reading["unit"] == "kg"
    assert reading["measured_on"] == date.today().isoformat()
    assert client.get("/me/readings", headers=auth(token)).json() == [reading]


def test_blood_pressure_keeps_both_numbers(client):
    r = _add(client, patient(client), kind="blood_pressure", value=120, value2=80)
    assert r.status_code == 201
    assert (r.json()["value"], r.json()["value2"], r.json()["unit"]) == (120, 80, "mmHg")


def test_readings_come_back_newest_measured_first(client):
    token = patient(client)
    _weight(client, token, 70, on=(date.today() - timedelta(days=5)).isoformat())
    _weight(client, token, 72, on=date.today().isoformat())
    _weight(client, token, 71, on=(date.today() - timedelta(days=2)).isoformat())

    assert [r["value"] for r in client.get("/me/readings", headers=auth(token)).json()] == [72, 71, 70]


def test_pulse_and_blood_pressure_are_whole_numbers(client):
    token = patient(client)
    assert _add(client, token, kind="pulse", value=72.6).json()["value"] == 73
    bp = _add(client, token, kind="blood_pressure", value=120.4, value2=79.6).json()
    assert (bp["value"], bp["value2"]) == (120, 80)


@pytest.mark.parametrize("body", [
    {"kind": "weight", "value": 0.5},
    {"kind": "weight", "value": 501},
    {"kind": "height", "value": 29},
    {"kind": "height", "value": 273},
    {"kind": "pulse", "value": 19},
    {"kind": "pulse", "value": 301},
    {"kind": "blood_pressure", "value": 39, "value2": 30},
    {"kind": "blood_pressure", "value": 301, "value2": 80},
    {"kind": "blood_pressure", "value": 120, "value2": 19},
    {"kind": "blood_pressure", "value": 250, "value2": 201},
    {"kind": "temperature", "value": 37},
])
def test_implausible_values_are_refused(client, body):
    assert _add(client, patient(client), **body).status_code == 422


@pytest.mark.parametrize("body", [
    {"kind": "weight", "value": 1},
    {"kind": "weight", "value": 500},
    {"kind": "height", "value": 30},
    {"kind": "height", "value": 272},
    {"kind": "pulse", "value": 20},
    {"kind": "pulse", "value": 300},
    {"kind": "blood_pressure", "value": 300, "value2": 200},
    {"kind": "blood_pressure", "value": 40, "value2": 20},
])
def test_the_limits_themselves_are_allowed(client, body):
    assert _add(client, patient(client), **body).status_code == 201


def test_blood_pressure_needs_both_numbers_in_order(client):
    token = patient(client)
    assert _add(client, token, kind="blood_pressure", value=120).status_code == 422
    assert _add(client, token, kind="blood_pressure", value=80, value2=120).status_code == 422
    assert _add(client, token, kind="blood_pressure", value=120, value2=120).status_code == 422


def test_a_second_number_is_refused_on_anything_but_blood_pressure(client):
    assert _add(client, patient(client), kind="weight", value=72, value2=80).status_code == 422


def test_a_date_more_than_a_day_ahead_is_refused(client):
    token = patient(client)
    too_far = (date.today() + timedelta(days=2)).isoformat()
    assert _add(client, token, kind="weight", value=72, measured_on=too_far).status_code == 422


def test_tomorrow_is_allowed_because_the_patient_may_be_ahead_of_the_server(client):
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    assert _add(client, patient(client), kind="weight", value=72, measured_on=tomorrow).status_code == 201


def test_a_reading_can_be_deleted(client):
    token = patient(client)
    reading = _weight(client, token)

    assert client.delete(f"/me/readings/{reading['id']}", headers=auth(token)).status_code == 204

    assert client.get("/me/readings", headers=auth(token)).json() == []


def test_another_patients_readings_are_invisible_and_untouchable(client):
    alice = patient(client, email="alice@example.com", name="Alice")
    bob = patient(client, email="bob@example.com", name="Bob")
    reading = _weight(client, alice)

    assert client.get("/me/readings", headers=auth(bob)).json() == []
    assert client.delete(f"/me/readings/{reading['id']}", headers=auth(bob)).status_code == 404
    assert len(client.get("/me/readings", headers=auth(alice)).json()) == 1


def test_readings_are_a_patient_feature_for_signed_in_users(client):
    assert client.get("/me/readings").status_code == 401
    assert client.get("/me/readings", headers=auth(doctor(client))).status_code == 403


def test_readings_are_in_the_export_and_go_with_the_account(client, db):
    from app.db.models import HealthReading
    token = patient(client)
    _weight(client, token)

    assert client.get("/me/export", headers=auth(token)).json()["readings"][0]["value"] == 72

    assert client.request("DELETE", "/me", json={"password": PASSWORD}, headers=auth(token)).status_code == 204
    assert db.query(HealthReading).count() == 0


# ── what the agents are told ─────────────────────────────────────────────────────────

def _background(client, graph, token):
    client.post("/process", json={"text": "I have a question"}, headers=auth(token))
    return graph.inputs[-1]["patient_background"]


def test_a_patient_with_no_readings_gets_the_same_background_as_before(client, graph):
    assert _background(client, graph, patient(client)) is None


def test_the_profile_lines_are_unchanged_by_having_no_readings(client, graph):
    token = patient(client)
    client.put("/me/profile", json={"gender": "female", "allergies": [], "medications": [],
                                    "conditions": []}, headers=auth(token))

    assert _background(client, graph, token) == (
        "Stated by the patient in their account profile:\n- Age: not provided\n- Gender: female\n"
        "- Allergies: none listed\n- Regular medications: none listed\n- Long-term conditions: none listed")


def test_only_the_latest_of_each_reading_reaches_the_agents(client, graph):
    token = patient(client)
    _weight(client, token, 70, on="2026-01-05")
    _weight(client, token, 72, on="2026-11-03")
    _add(client, token, kind="blood_pressure", value=120, value2=80, measured_on="2026-11-08")
    _add(client, token, kind="pulse", value=72, measured_on="2026-11-08")
    _add(client, token, kind="height", value=181, measured_on="2026-02-01")

    assert _background(client, graph, token) == (
        "Recorded by the patient on their dashboard:\n"
        "- Latest weight: 72 kg (recorded 3 Nov 2026)\n"
        "- Latest height: 181 cm (recorded 1 Feb 2026)\n"
        "- Latest blood pressure: 120/80 mmHg (recorded 8 Nov 2026)\n"
        "- Latest pulse: 72 bpm (recorded 8 Nov 2026)")


def test_a_decimal_weight_keeps_its_decimal(client, graph):
    token = patient(client)
    _weight(client, token, 72.5, on="2026-11-03")
    assert "72.5 kg" in _background(client, graph, token)


def test_a_reviewing_doctor_sees_the_readings_the_agents_saw(client, graph):
    doc = doctor(client)
    token = patient(client)
    _weight(client, token, 72, on="2026-11-03")
    graph.level = "level_2"
    client.post("/process", json={"text": "I have a question"}, headers=auth(token))
    [case] = client.get("/doctor/cases", headers=auth(doc)).json()

    detail = client.get(f"/doctor/cases/{case['id']}", headers=auth(doc)).json()

    assert "- Latest weight: 72 kg (recorded 3 Nov 2026)" in detail["background"]


def test_child_mode_never_sees_the_parents_readings(client, child_graph):
    token = patient(client)
    _weight(client, token)

    client.post("/process", json={"text": "My baby has a fever", "mode": "child"}, headers=auth(token))

    assert "patient_background" not in child_graph.inputs[-1]


# ── the follow-up calendar ───────────────────────────────────────────────────────────

def _episode(client, graph, token, diagnosis="Viral pharyngitis"):
    graph.diagnosis = diagnosis
    r = client.post("/process", json={"text": "My throat hurts"}, headers=auth(token))
    assert r.status_code == 200, r.text
    return r.json()


def _calendar(client, token):
    r = client.get("/me/followups", headers=auth(token))
    assert r.status_code == 200, r.text
    return r.json()


def _kinds(episode):
    return [e["kind"] for e in episode["entries"]]


def test_a_patient_with_no_follow_ups_has_an_empty_calendar(client):
    assert _calendar(client, patient(client)) == []


def test_a_new_episode_shows_the_next_check_in_and_the_expected_one_after_it(client, graph):
    token = patient(client)
    turn = _episode(client, graph, token)

    [episode] = _calendar(client, token)

    assert episode["conversation_id"] == turn["conversation_id"]
    assert episode["label"] == "Sore throat"
    assert _kinds(episode) == ["upcoming", "expected"]
    upcoming, expected = (datetime.fromisoformat(e["at"]) for e in episode["entries"])
    assert upcoming == datetime.fromisoformat(episode["next_checkin_at"])
    # Sore throat is checked at day 3 and day 7, so the second comes four days after the first.
    assert expected - upcoming == timedelta(days=4)


def test_a_due_check_in_shows_as_due_with_the_next_still_expected(client, graph):
    token = patient(client)
    _episode(client, graph, token)

    followup.run_due(now=_at(4))

    [episode] = _calendar(client, token)
    assert _kinds(episode) == ["due", "expected"]
    assert episode["checkin_due"] is True


def test_an_answered_check_in_records_the_answer_and_the_next_is_now_fixed(client, graph):
    token = patient(client)
    _episode(client, graph, token)
    followup.run_due(now=_at(4))
    [episode] = _calendar(client, token)

    client.post(f"/followups/{episode['id']}/checkin", json={"overall": "same", "signs": []}, headers=auth(token))

    [episode] = _calendar(client, token)
    assert _kinds(episode) == ["answered", "upcoming"]
    assert episode["entries"][0]["overall"] == "same"


def test_silence_shows_as_missed_and_never_as_answered(client, graph):
    token = patient(client)
    _episode(client, graph, token)
    for day in (4, 5, 6):
        followup.run_due(now=_at(day))

    [episode] = _calendar(client, token)

    assert episode["state"] == "lapsed"
    assert _kinds(episode) == ["missed"]


def test_a_resolved_episode_has_nothing_left_to_come(client, graph):
    token = patient(client)
    _episode(client, graph, token)
    followup.run_due(now=_at(4))
    [episode] = _calendar(client, token)

    client.post(f"/followups/{episode['id']}/checkin", json={"overall": "better", "signs": []}, headers=auth(token))

    [episode] = _calendar(client, token)
    assert episode["state"] == "resolved"
    assert _kinds(episode) == ["answered"]


def test_a_postponed_check_in_moves_rather_than_counting(client, graph):
    token = patient(client)
    _episode(client, graph, token)
    followup.run_due(now=_at(4))
    [episode] = _calendar(client, token)

    client.post(f"/followups/{episode['id']}/postpone", json={"hours": 24}, headers=auth(token))

    [episode] = _calendar(client, token)
    assert _kinds(episode) == ["upcoming", "expected"]


def test_a_stopped_episode_is_shown_without_a_recovery(client, graph):
    token = patient(client)
    _episode(client, graph, token)
    [episode] = _calendar(client, token)

    client.post(f"/followups/{episode['id']}/stop", headers=auth(token))

    [episode] = _calendar(client, token)
    assert episode["state"] == "stopped"
    assert _kinds(episode) == []


def test_a_one_check_in_plan_expects_nothing_further(client, graph):
    token = patient(client)
    _episode(client, graph, token, diagnosis="Acute bronchitis")

    [episode] = _calendar(client, token)

    assert episode["complaint"] == "cough"
    assert _kinds(episode) == ["upcoming"]


def test_the_newest_episode_comes_first_and_others_are_private(client, graph):
    token = patient(client)
    first = _episode(client, graph, token)
    graph.title = "Second"
    client.post(f"/followups/{_calendar(client, token)[0]['id']}/stop", headers=auth(token))
    second = _episode(client, graph, token, diagnosis="Acute bronchitis")
    other = patient(client, email="other@example.com", name="Other")

    assert [e["conversation_id"] for e in _calendar(client, token)] == [second["conversation_id"], first["conversation_id"]]
    assert _calendar(client, other) == []


def test_the_calendar_is_a_patient_feature(client):
    assert client.get("/me/followups", headers=auth(doctor(client))).status_code == 403
