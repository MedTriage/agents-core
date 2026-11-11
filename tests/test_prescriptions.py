"""A prescription's life: the prescriber drafts it on a level_2 turn, a doctor edits and issues
it, and only then can the patient see it. The rule these tests exist for: a draft never reaches
a patient unless the test switch is on."""
import json

import pytest

from app.services import prescriptions
from tests.conftest import auth, doctor, patient

RX = {
    "diagnosis": "Viral pharyngitis",
    "medications": [{"drug": "paracetamol", "dose": "1 g", "frequency": "every 6 hours",
                     "duration": "3 days", "route": "oral", "notes": ""}],
    "refer_now": False,
    "investigations": ["Throat swab"],
    "advice": ["Rest and drink fluids."],
    "return_precautions": ["Trouble breathing."],
    "assumptions": [],
}
MED = RX["medications"][0]


def _held(client, graph, token, rx=RX, problems=()):
    """A level_2 turn with what the prescriber came back with."""
    graph.level = "level_2"
    graph.prescription = rx
    graph.prescription_problems = list(problems)
    r = client.post("/process", json={"text": "My throat is very sore"}, headers=auth(token))
    assert r.status_code == 200, r.text
    return r.json()


def _case_id(client, doc):
    return client.get("/doctor/cases", headers=auth(doc)).json()[0]["id"]


def _review(client, doc, case_id, status="verified", **body):
    return client.patch(f"/doctor/cases/{case_id}", json={"status": status, **body}, headers=auth(doc))


def _mine(client, token):
    r = client.get("/me/prescriptions", headers=auth(token))
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def switch_on(monkeypatch):
    monkeypatch.setattr(prescriptions, "PRESCRIPTION_DEMO_INSTANT", True)


# ── a draft goes to the doctor ───────────────────────────────────────────────────────

def test_a_level_2_turn_keeps_the_draft_for_the_reviewing_doctor(client, graph):
    doc = doctor(client)
    _held(client, graph, patient(client))

    view = client.get(f"/doctor/cases/{_case_id(client, doc)}", headers=auth(doc)).json()["prescription"]

    assert view["status"] == "draft"
    assert view["draft"] == RX
    assert view["problems"] == []


def test_a_held_case_with_no_draft_shows_the_doctor_no_prescription(client, graph):
    doc = doctor(client)
    graph.level = "level_2"
    client.post("/process", json={"text": "My throat is very sore"}, headers=auth(patient(client)))

    assert client.get(f"/doctor/cases/{_case_id(client, doc)}", headers=auth(doc)).json()["prescription"] is None


def test_a_draft_that_failed_validation_reaches_the_doctor_with_its_problems(client, graph):
    doc = doctor(client)
    _held(client, graph, patient(client), problems=["medication 1 (paracetamol): 'dose' must be specific"])

    view = client.get(f"/doctor/cases/{_case_id(client, doc)}", headers=auth(doc)).json()["prescription"]

    assert view["problems"] == ["medication 1 (paracetamol): 'dose' must be specific"]


# ── a draft never reaches the patient ────────────────────────────────────────────────

def test_the_turn_payload_carries_no_draft(client, graph):
    payload = _held(client, graph, patient(client))

    assert payload["prescription"] is None
    assert "prescription_attempts" not in payload and "prescription_problems" not in payload
    assert "paracetamol" not in json.dumps(payload)


def test_the_stream_carries_no_draft(client, graph):
    graph.level = "level_2"
    graph.prescription = RX
    r = client.post("/process/stream", json={"text": "My throat is very sore"},
                    headers=auth(patient(client)))

    assert r.status_code == 200
    assert '"type": "done"' in r.text or '"type":"done"' in r.text
    assert "paracetamol" not in r.text


def test_conversation_detail_carries_no_draft(client, graph):
    token = patient(client)
    conversation_id = _held(client, graph, token)["conversation_id"]

    detail = client.get(f"/conversations/{conversation_id}", headers=auth(token)).json()

    assert detail["prescriptions"] == []
    assert "paracetamol" not in json.dumps(detail)


def test_the_patients_list_and_export_carry_no_draft(client, graph):
    token = patient(client)
    _held(client, graph, token)

    assert _mine(client, token) == []
    assert "paracetamol" not in json.dumps(client.get("/me/export", headers=auth(token)).json())


def test_a_draft_cannot_be_opened_by_its_id(client, graph, db):
    from app.db.models import Prescription
    token = patient(client)
    _held(client, graph, token)
    draft_id = db.query(Prescription).one().id

    assert client.get(f"/me/prescriptions/{draft_id}", headers=auth(token)).status_code == 404


# ── the doctor edits and issues ──────────────────────────────────────────────────────

def test_approving_with_edits_issues_the_doctors_version(client, graph):
    doc = doctor(client, registration="WBMC-12345")
    token = patient(client)
    _held(client, graph, token)
    edited = {**RX, "medications": [{**MED, "dose": "500 mg"}]}

    r = _review(client, doc, _case_id(client, doc), prescription=edited)

    assert r.status_code == 200, r.text
    [issued] = _mine(client, token)
    assert issued["status"] == "issued"
    assert issued["medications"][0]["dose"] == "500 mg"
    assert issued["issued_by_name"] == "Dr Rao"
    assert issued["issued_by_registration"] == "WBMC-12345"
    assert issued["issued_at"]


def test_approving_without_edits_issues_the_draft(client, graph):
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)

    assert _review(client, doc, _case_id(client, doc)).status_code == 200

    assert _mine(client, token)[0]["medications"][0]["dose"] == "1 g"


def test_deleting_every_row_issues_nothing_but_still_approves_the_answer(client, graph):
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)

    r = _review(client, doc, _case_id(client, doc), prescription={**RX, "medications": []})

    assert r.status_code == 200 and r.json()["status"] == "verified"
    assert _mine(client, token) == []


def test_rejecting_the_case_discards_the_draft(client, graph):
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)
    case_id = _case_id(client, doc)

    assert _review(client, doc, case_id, status="rejected").status_code == 200

    assert _mine(client, token) == []
    assert client.get(f"/doctor/cases/{case_id}", headers=auth(doc)).json()["prescription"]["status"] == "discarded"


def test_a_doctor_can_write_a_prescription_when_there_was_no_draft(client, graph):
    doc = doctor(client)
    token = patient(client)
    graph.level = "level_2"
    client.post("/process", json={"text": "My throat is very sore"}, headers=auth(token))

    assert _review(client, doc, _case_id(client, doc), prescription=RX).status_code == 200

    assert _mine(client, token)[0]["diagnosis"] == "Viral pharyngitis"


def test_a_draft_with_problems_cannot_be_issued_without_a_doctors_edit(client, graph):
    doc = doctor(client)
    token = patient(client)
    incomplete = {**RX, "medications": [{**MED, "dose": ""}]}
    _held(client, graph, token, rx=incomplete, problems=["medication 1 (paracetamol): 'dose' must be specific"])
    case_id = _case_id(client, doc)

    assert _review(client, doc, case_id).status_code == 422

    assert client.get(f"/doctor/cases/{case_id}", headers=auth(doc)).json()["status"] == "pending"
    assert _mine(client, token) == []


@pytest.mark.parametrize("field", ["drug", "dose", "frequency", "duration", "route"])
def test_the_doctors_rows_need_every_field(client, graph, field):
    doc = doctor(client)
    _held(client, graph, patient(client))

    r = _review(client, doc, _case_id(client, doc),
                prescription={**RX, "medications": [{**MED, field: "  "}]})

    assert r.status_code == 422


def test_a_prescription_needs_a_diagnosis(client, graph):
    doc = doctor(client)
    _held(client, graph, patient(client))

    assert _review(client, doc, _case_id(client, doc), prescription={**RX, "diagnosis": ""}).status_code == 422


# ── allergies ────────────────────────────────────────────────────────────────────────

def _allergic_patient(client):
    token = patient(client)
    client.put("/me/profile", json={"allergies": [{"name": "penicillin", "detail": "rash"}],
                                    "medications": [], "conditions": []}, headers=auth(token))
    return token


def test_a_drug_that_conflicts_with_a_stated_allergy_is_refused(client, graph):
    doc = doctor(client)
    token = _allergic_patient(client)
    _held(client, graph, token)
    case_id = _case_id(client, doc)

    r = _review(client, doc, case_id, prescription={**RX, "medications": [{**MED, "drug": "amoxicillin"}]})

    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "allergy_conflict"
    assert r.json()["detail"]["conflicts"] == [{"drug": "amoxicillin", "allergy": "penicillin"}]
    assert client.get(f"/doctor/cases/{case_id}", headers=auth(doc)).json()["status"] == "pending"
    assert _mine(client, token) == []


def test_the_doctor_can_confirm_past_an_allergy_conflict(client, graph):
    doc = doctor(client)
    token = _allergic_patient(client)
    _held(client, graph, token)

    r = _review(client, doc, _case_id(client, doc), confirm_allergy=True,
                prescription={**RX, "medications": [{**MED, "drug": "amoxicillin"}]})

    assert r.status_code == 200
    assert _mine(client, token)[0]["medications"][0]["drug"] == "amoxicillin"


def test_an_allergy_added_after_the_turn_is_still_checked(client, graph):
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)
    client.put("/me/profile", json={"allergies": [{"name": "penicillin", "detail": None}],
                                    "medications": [], "conditions": []}, headers=auth(token))

    r = _review(client, doc, _case_id(client, doc), prescription={**RX, "medications": [{**MED, "drug": "amoxicillin"}]})

    assert r.status_code == 409


# ── "go to hospital now" on a held case ──────────────────────────────────────────────

def test_a_hospital_flag_puts_the_case_first_and_changes_no_level(client, graph):
    doc = doctor(client)
    first = patient(client, email="first@example.com", name="First")
    second = patient(client, email="second@example.com", name="Second")
    _held(client, graph, first)
    payload = _held(client, graph, second, rx={**RX, "refer_now": True})

    cases = client.get("/doctor/cases", headers=auth(doc)).json()

    assert payload["triage_level"] == "level_2"
    assert [(c["patient_name"], c["hospital_flag"]) for c in cases] == [("Second", True), ("First", False)]


# ── what the patient gets ────────────────────────────────────────────────────────────

def test_the_issued_sheet_has_what_the_design_shows(client, graph):
    doc = doctor(client, registration="WBMC-12345")
    token = patient(client)
    _held(client, graph, token)
    _review(client, doc, _case_id(client, doc))
    rx_id = _mine(client, token)[0]["id"]

    sheet = client.get(f"/me/prescriptions/{rx_id}", headers=auth(token)).json()

    assert sheet["patient_message"] == "My throat is very sore"
    assert sheet["answer"] == "Rest and drink fluids."
    assert sheet["diagnosis"] == "Viral pharyngitis"
    assert sheet["medications"] == [MED]
    assert sheet["investigations"] == ["Throat swab"]
    assert sheet["advice"] == ["Rest and drink fluids."]
    assert sheet["return_precautions"] == ["Trouble breathing."]
    assert sheet["issued_by_registration"] == "WBMC-12345"


def test_conversation_detail_links_the_issued_prescription_to_its_answer(client, graph):
    doc = doctor(client)
    token = patient(client)
    turn = _held(client, graph, token)
    _review(client, doc, _case_id(client, doc))

    detail = client.get(f"/conversations/{turn['conversation_id']}", headers=auth(token)).json()

    [link] = detail["prescriptions"]
    assert link["message_id"] == turn["message_id"]
    assert link["status"] == "issued"
    assert link["id"] == _mine(client, token)[0]["id"]


def test_another_patients_prescription_is_not_found(client, graph):
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)
    _review(client, doc, _case_id(client, doc))
    rx_id = _mine(client, token)[0]["id"]
    other = patient(client, email="other@example.com", name="Other")

    assert client.get(f"/me/prescriptions/{rx_id}", headers=auth(other)).status_code == 404
    assert _mine(client, other) == []


def test_a_malformed_id_is_not_found(client):
    assert client.get("/me/prescriptions/not-a-uuid", headers=auth(patient(client))).status_code == 404


def test_a_doctor_has_no_patient_prescriptions(client):
    assert client.get("/me/prescriptions", headers=auth(doctor(client))).status_code == 403


def test_the_sheet_survives_the_doctors_account_being_deleted(client, graph):
    from tests.conftest import PASSWORD
    doc = doctor(client, registration="WBMC-12345")
    token = patient(client)
    _held(client, graph, token)
    _review(client, doc, _case_id(client, doc))

    assert client.request("DELETE", "/me", json={"password": PASSWORD}, headers=auth(doc)).status_code == 204

    assert _mine(client, token)[0]["issued_by_name"] == "Dr Rao"


def test_an_issued_prescription_is_in_the_export(client, graph):
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)
    _review(client, doc, _case_id(client, doc))

    export = client.get("/me/export", headers=auth(token)).json()

    assert export["prescriptions"][0]["medications"] == [MED]


def test_an_issued_prescription_is_deleted_with_the_patient(client, graph, db):
    from app.db.models import Prescription
    from tests.conftest import PASSWORD
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)
    _review(client, doc, _case_id(client, doc))

    assert client.request("DELETE", "/me", json={"password": PASSWORD}, headers=auth(token)).status_code == 204

    assert db.query(Prescription).count() == 0


# ── what the agents are told afterwards ──────────────────────────────────────────────

def test_an_issued_prescription_reaches_the_next_turns_background_but_not_the_profile(client, graph):
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)
    _review(client, doc, _case_id(client, doc))

    graph.level = "level_1"
    client.post("/process", json={"text": "Another question"}, headers=auth(token))

    background = graph.inputs[-1]["patient_background"]
    assert "Prescribed by a doctor through MedTriage:" in background
    assert "paracetamol 1 g, oral, every 6 hours, for 3 days" in background
    assert "by Dr Rao" in background
    assert client.get("/me/profile", headers=auth(token)).json()["medications"] == []


def test_a_draft_never_reaches_the_background(client, graph):
    token = patient(client)
    _held(client, graph, token)

    graph.level = "level_1"
    client.post("/process", json={"text": "Another question"}, headers=auth(token))

    assert "paracetamol" not in (graph.inputs[-1]["patient_background"] or "")


# ── the test switch ──────────────────────────────────────────────────────────────────

def test_with_the_switch_on_the_turn_links_the_draft_at_once(client, graph, switch_on):
    turn = _held(client, graph, patient(client))

    assert turn["prescription"]["status"] == "draft"
    assert turn["prescription"]["message_id"] == turn["message_id"]
    # Only the link: the draft's contents come from the sheet, which says what it is.
    assert "paracetamol" not in json.dumps(turn["prescription"])


def test_with_the_switch_on_the_patient_can_open_the_draft_without_the_held_answer(client, graph, switch_on):
    token = patient(client)
    turn = _held(client, graph, token)

    sheet = client.get(f"/me/prescriptions/{turn['prescription']['id']}", headers=auth(token)).json()

    assert sheet["status"] == "draft"
    assert sheet["medications"] == [MED]
    assert sheet["answer"] is None
    assert sheet["issued_by_name"] is None


def test_with_the_switch_on_the_answer_stays_held_for_the_doctor(client, graph, switch_on):
    doc = doctor(client)
    _held(client, graph, patient(client))

    assert client.get("/doctor/cases", headers=auth(doc)).json()[0]["status"] == "pending"


def test_with_the_switch_on_a_rejected_draft_disappears(client, graph, switch_on):
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)
    assert len(_mine(client, token)) == 1

    _review(client, doc, _case_id(client, doc), status="rejected")

    assert _mine(client, token) == []


def test_with_the_switch_on_an_issued_prescription_replaces_its_draft(client, graph, switch_on):
    doc = doctor(client)
    token = patient(client)
    _held(client, graph, token)

    _review(client, doc, _case_id(client, doc), prescription={**RX, "medications": [{**MED, "dose": "500 mg"}]})

    [only] = _mine(client, token)
    assert only["status"] == "issued"
    assert only["medications"][0]["dose"] == "500 mg"


def test_with_the_switch_on_a_draft_stays_out_of_the_background_and_export(client, graph, switch_on):
    token = patient(client)
    _held(client, graph, token)

    graph.level = "level_1"
    client.post("/process", json={"text": "Another question"}, headers=auth(token))

    assert "paracetamol" not in (graph.inputs[-1]["patient_background"] or "")
    assert "paracetamol" not in json.dumps(client.get("/me/export", headers=auth(token)).json())


def test_the_switch_is_off_by_default():
    from app import config
    assert config.PRESCRIPTION_DEMO_INSTANT is False


def test_the_server_warns_at_startup_when_the_switch_is_on(monkeypatch, caplog, graph, child_graph):
    from fastapi.testclient import TestClient
    from app import main

    monkeypatch.setattr(main, "PRESCRIPTION_DEMO_INSTANT", True)
    main.app.state.graph, main.app.state.child_graph = graph, child_graph

    with caplog.at_level("WARNING"), TestClient(main.app):
        pass

    assert any("PRESCRIPTION_DEMO_INSTANT is on" in record.message for record in caplog.records)
