"""Follow-up mode: a level_1 answer (or a level_2 answer a doctor approved) becomes a watched
episode with "come back if" signs and scheduled check-ins. Silence is never recovery."""
import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.db.session import engine
from app.services import followup
from app.services.followup_content import classify, plan_for
from tests.conftest import auth, doctor, outbox, patient

IST = ZoneInfo("Asia/Kolkata")


def _sql(query, **params):
    with engine.begin() as conn:
        return conn.execute(text(query), params)


def _at(days: int, hour: int = 10) -> datetime:
    """`days` from now at `hour`:00 India time, outside the quiet hours unless asked."""
    local = (datetime.now(IST) + timedelta(days=days)).replace(hour=hour, minute=0, second=0, microsecond=0)
    return local.astimezone(timezone.utc)


def _turn(client, token, text="My throat hurts", conversation_id=None):
    body = {"text": text, **({"conversation_id": conversation_id} if conversation_id else {})}
    r = client.post("/process", json=body, headers=auth(token))
    assert r.status_code == 200, r.text
    return r.json()


def _episode(client, token, graph, diagnosis="Viral pharyngitis"):
    graph.diagnosis = diagnosis
    turn = _turn(client, token)
    assert turn["followup"] is not None
    return turn["conversation_id"], turn["followup"]


def _checkin(client, token, episode_id, overall, signs=(), new=None):
    return client.post(f"/followups/{episode_id}/checkin",
                       json={"overall": overall, "signs": list(signs), "new": new}, headers=auth(token))


def _detail(client, token, conversation_id):
    return client.get(f"/conversations/{conversation_id}", headers=auth(token)).json()


# ── content: NICE timing and signs ───────────────────────────────────────────────────

def test_complaints_are_recognised_from_the_diagnosis_or_the_words_used():
    assert classify("Viral pharyngitis") == "sore_throat"
    assert classify("Acute bronchitis") == "cough"
    assert classify("Acute otitis media") == "ear_infection"
    assert classify("Acute rhinosinusitis") == "sinusitis"
    assert classify("Lower urinary tract infection") == "uti"
    assert classify("Common cold", "my nose is blocked") == "general"
    assert classify("", "it burns when I pee, probably a UTI") == "uti"


def test_check_ins_follow_nice_timing_for_each_complaint():
    assert plan_for("sore_throat").checkin_days == (3, 7)
    assert plan_for("ear_infection").checkin_days == (3, 7)
    assert plan_for("sinusitis").checkin_days == (7, 21)
    assert plan_for("cough").checkin_days == (21,)
    assert plan_for("uti").checkin_days == (2,)
    assert plan_for("general").checkin_days == (3, 7)


def test_every_plan_carries_the_general_signs_and_its_own():
    general = {s["text"] for s in plan_for("general").return_signs}
    sinus = plan_for("sinusitis").return_signs
    assert general <= {s["text"] for s in sinus}
    assert any("eye" in s["text"].lower() and s["level"] == "level_3" for s in sinus)
    assert any("fever" in s["text"].lower() and s["level"] == "level_2" for s in plan_for("uti").return_signs)


def test_children_under_five_also_get_the_who_return_immediately_signs():
    adult = plan_for("cough", age=30).return_signs
    child = plan_for("cough", age=3).return_signs
    assert not any("breastfeed" in s["text"].lower() for s in adult)
    assert any("breastfeed" in s["text"].lower() and s["level"] == "level_3" for s in child)


# ── naming the complaint: Jev, else the word list ────────────────────────────────────

def _record(*symptoms, conditions=()):
    return {"current_conditions": list(conditions), "past_conditions": [], "medications": [],
            "allergies": [], "demographics": {}, "notes": [], "turn": 1,
            "symptoms": [{"text": s, "turn": 1} for s in symptoms]}


def test_jevs_confident_choice_names_the_complaint(client, graph, jev):
    jev.probabilities = {"uti": 0.93, "general": 0.05, "cough": 0.02}
    graph.diagnosis = "Viral illness"

    turn = _turn(client, patient(client), "it burns when I wee")

    assert turn["followup"]["complaint"] == "uti"


def test_an_unsure_jev_gets_the_general_plan(client, graph, jev):
    jev.probabilities = {"cough": 0.6, "general": 0.4}
    graph.diagnosis = "Acute bronchitis"

    turn = _turn(client, patient(client))

    assert turn["followup"]["complaint"] == "general"


def test_jev_is_sent_the_diagnosis_and_the_record_and_nothing_about_who_the_patient_is(client, graph, jev):
    jev.probabilities = {"general": 0.9}
    graph.diagnosis = "Viral illness"
    graph.scribe_record = _record("fever", "tiredness", conditions=["asthma"])

    _turn(client, patient(client), "I have a fever and feel tired, no cough")

    assert jev.calls[-1]["state"] == {"diagnosis": "Viral illness", "conditions": ["asthma"],
                                      "symptoms": ["fever", "tiredness"]}


def test_without_jev_the_word_list_reads_the_record_not_the_raw_message(client, graph):
    graph.diagnosis = "Viral illness"
    graph.scribe_record = _record("fever", "tiredness")

    turn = _turn(client, patient(client), "I have a fever and feel tired, no cough")

    assert turn["followup"]["complaint"] == "general"  # not "cough", from "no cough"


# ── when an episode opens ────────────────────────────────────────────────────────────

def test_a_level_1_answer_opens_an_episode_with_its_signs_and_first_check_in(client, graph):
    token = patient(client)
    before = datetime.now(timezone.utc)

    conversation_id, episode = _episode(client, token, graph, diagnosis="Viral pharyngitis")

    assert episode["state"] == "active"
    assert episode["complaint"] == "sore_throat"
    assert episode["return_signs"] and all(s["level"] in ("level_2", "level_3") for s in episode["return_signs"])
    due = datetime.fromisoformat(episode["next_checkin_at"])
    assert timedelta(days=3) <= due - before < timedelta(days=3, minutes=5)
    assert _detail(client, token, conversation_id)["followup"]["id"] == episode["id"]


def test_chitchat_and_clarification_questions_open_no_episode(client, graph):
    token = patient(client)
    graph.intent = "chitchat"
    assert _turn(client, token, "hello")["followup"] is None
    graph.intent, graph.clarification = "clinical_query", True
    assert _turn(client, token, "I feel off")["followup"] is None


def test_a_level_3_answer_opens_no_episode(client, graph):
    token = patient(client)
    graph.level = "level_3"
    assert _turn(client, token)["followup"] is None


def test_a_level_2_answer_opens_an_episode_only_once_a_doctor_approves_it(client, graph):
    doc = doctor(client)
    token = patient(client)
    graph.level = "level_2"
    turn = _turn(client, token)
    assert turn["followup"] is None
    [case] = client.get("/doctor/cases", headers=auth(doc)).json()

    client.patch(f"/doctor/cases/{case['id']}", json={"status": "verified", "notes": "ok"}, headers=auth(doc))

    episode = _detail(client, token, turn["conversation_id"])["followup"]
    assert episode is not None and episode["state"] == "active"


def test_a_rejected_level_2_answer_opens_no_episode(client, graph):
    doc = doctor(client)
    token = patient(client)
    graph.level = "level_2"
    turn = _turn(client, token)
    [case] = client.get("/doctor/cases", headers=auth(doc)).json()

    client.patch(f"/doctor/cases/{case['id']}", json={"status": "rejected", "notes": "see me"}, headers=auth(doc))

    assert _detail(client, token, turn["conversation_id"])["followup"] is None


def test_a_conversation_has_one_active_episode_at_a_time(client, graph):
    token = patient(client)
    conversation_id, episode = _episode(client, token, graph)

    later = _turn(client, token, "still sore", conversation_id=conversation_id)

    assert later["followup"]["id"] == episode["id"]
    assert _sql("SELECT count(*) FROM episodes").scalar_one() == 1


def test_the_demo_toggle_makes_every_wait_a_few_seconds(client, graph, monkeypatch):
    monkeypatch.setattr(followup, "FOLLOWUP_DEMO_SECONDS", 60)
    token = patient(client)
    before = datetime.now(timezone.utc)

    conversation_id, episode = _episode(client, token, graph)

    due = datetime.fromisoformat(episode["next_checkin_at"])
    assert timedelta(seconds=60) <= due - before < timedelta(seconds=90)
    # Quiet hours do not apply in demo mode: a run at 23:00 still sends the check-in.
    followup.run_due(now=_at(0, hour=23) + timedelta(days=1))
    assert len(outbox("checkin_due")) == 1


# ── the scheduler ────────────────────────────────────────────────────────────────────

def test_nothing_happens_before_a_check_in_is_due(client, graph):
    token = patient(client)
    _episode(client, token, graph)

    followup.run_due(now=_at(1))

    assert outbox("checkin_due") == []


def test_a_due_check_in_is_opened_and_announced_without_health_details(client, graph):
    token = patient(client)
    graph.response = "This looks like viral pharyngitis."
    conversation_id, _ = _episode(client, token, graph)

    followup.run_due(now=_at(4))

    [email] = outbox("checkin_due", to="pat@example.com")
    flat = json.dumps(email["payload"]).lower()
    for health_word in ("throat", "pharyngitis", "sore"):
        assert health_word not in flat
    assert _detail(client, token, conversation_id)["followup"]["checkin_due"] is True
    [listed] = client.get("/conversations", headers=auth(token)).json()
    assert listed["checkin_due"] is True


def test_nothing_is_sent_during_quiet_hours(client, graph):
    token = patient(client)
    _episode(client, token, graph)

    followup.run_due(now=_at(4, hour=23))
    assert outbox("checkin_due") == []

    followup.run_due(now=_at(5, hour=9))
    assert len(outbox("checkin_due")) == 1


def test_silence_gets_one_reminder_then_lapses_and_is_never_counted_as_recovery(client, graph):
    token = patient(client)
    conversation_id, _ = _episode(client, token, graph)

    followup.run_due(now=_at(4))
    followup.run_due(now=_at(5))
    followup.run_due(now=_at(6))
    followup.run_due(now=_at(9))

    reminders = [e for e in outbox("checkin_due") if e["payload"]["reminder"]]
    assert len(outbox("checkin_due")) == 2 and len(reminders) == 1
    assert _detail(client, token, conversation_id)["followup"]["state"] == "lapsed"


def test_a_lapsed_doctor_approved_episode_goes_back_to_that_doctor_only(client, graph):
    doc_a = doctor(client, email="a@example.com")
    doc_b = doctor(client, email="b@example.com")
    token = patient(client)
    graph.level = "level_2"
    _turn(client, token)
    [case] = client.get("/doctor/cases", headers=auth(doc_a)).json()
    client.patch(f"/doctor/cases/{case['id']}", json={"status": "verified", "notes": "ok"}, headers=auth(doc_a))

    for day in (4, 5, 6):
        followup.run_due(now=_at(day))

    assert len(outbox("followup_unanswered", to="a@example.com")) == 1
    assert outbox("followup_unanswered", to="b@example.com") == []
    [unanswered] = client.get("/doctor/followups", headers=auth(doc_a)).json()
    assert unanswered["patient_name"] == "Pat"
    assert client.get("/doctor/followups", headers=auth(doc_b)).json() == []
    assert client.get("/doctor/followups", headers=auth(token)).status_code == 403


# ── answering a check-in ─────────────────────────────────────────────────────────────

def _due_episode(client, token, graph, diagnosis="Viral pharyngitis", days=4):
    conversation_id, episode = _episode(client, token, graph, diagnosis=diagnosis)
    followup.run_due(now=_at(days))
    return conversation_id, episode


def test_better_closes_the_episode(client, graph):
    token = patient(client)
    conversation_id, episode = _due_episode(client, token, graph)

    r = _checkin(client, token, episode["id"], "better")

    assert r.json()["outcome"] == "resolved"
    detail = _detail(client, token, conversation_id)
    assert detail["followup"]["state"] == "resolved"
    assert [m["role"] for m in detail["messages"]][-2:] == ["user", "assistant"]


def test_the_same_at_an_early_check_in_schedules_the_next_one(client, graph):
    token = patient(client)
    conversation_id, episode = _due_episode(client, token, graph)
    answered = datetime.now(timezone.utc)

    r = _checkin(client, token, episode["id"], "same")

    assert r.json()["outcome"] == "scheduled"
    nxt = datetime.fromisoformat(_detail(client, token, conversation_id)["followup"]["next_checkin_at"])
    assert timedelta(days=4) <= nxt - answered < timedelta(days=4, minutes=5)


def test_the_same_at_the_last_check_in_goes_to_a_doctor(client, graph):
    doctor(client)
    token = patient(client)
    _, episode = _due_episode(client, token, graph, diagnosis="Lower urinary tract infection", days=3)

    r = _checkin(client, token, episode["id"], "same")

    assert r.json()["outcome"] == "doctor"
    assert r.json()["message"]["triage_level"] == "level_2"
    assert _sql("SELECT count(*) FROM review_cases").scalar_one() == 1
    assert len(outbox("cases_waiting")) == 1


def test_an_urgent_sign_locks_the_conversation_with_urgent_advice(client, graph):
    token = patient(client)
    conversation_id, episode = _due_episode(client, token, graph, diagnosis="Acute sinusitis", days=8)
    urgent = next(s["text"] for s in episode["return_signs"] if "eye" in s["text"].lower())

    r = _checkin(client, token, episode["id"], "worse", signs=[urgent])

    assert r.json()["outcome"] == "urgent"
    assert r.json()["message"]["triage_level"] == "level_3"
    detail = _detail(client, token, conversation_id)
    assert detail["locked"] is True and detail["followup"]["state"] == "escalated"


def test_a_level_2_sign_goes_to_a_doctor(client, graph):
    doctor(client)
    token = patient(client)
    _, episode = _due_episode(client, token, graph, diagnosis="Lower urinary tract infection", days=3)
    fever = next(s["text"] for s in episode["return_signs"] if "fever" in s["text"].lower())

    r = _checkin(client, token, episode["id"], "better", signs=[fever])

    assert r.json()["outcome"] == "doctor"
    assert _sql("SELECT count(*) FROM review_cases").scalar_one() == 1


def test_worse_sends_the_patient_back_through_triage_at_level_2_or_above(client, graph):
    doctor(client)
    token = patient(client)
    conversation_id, episode = _due_episode(client, token, graph)

    r = _checkin(client, token, episode["id"], "worse", new="now my ear hurts too")
    assert r.json()["outcome"] == "retriage"

    graph.level = "level_1"
    turn = _turn(client, token, "It's getting worse, and now my ear hurts", conversation_id=conversation_id)

    assert turn["triage_level"] == "level_2"
    assert turn["explanation"]["outcome"]["triage_level"] == 2
    assert "getting worse" in graph.inputs[-1]["patient_background"]
    assert _sql("SELECT count(*) FROM review_cases").scalar_one() == 1
    assert _detail(client, token, conversation_id)["followup"]["state"] == "escalated"


def test_better_with_something_new_sends_the_new_words_through_triage(client, graph):
    token = patient(client)
    conversation_id, episode = _due_episode(client, token, graph)

    r = _checkin(client, token, episode["id"], "better", new="now there is blood when I cough")

    assert r.json()["outcome"] == "reassess"
    assert r.json()["message"] is None  # the patient's next message carries the words
    assert _detail(client, token, conversation_id)["followup"]["state"] == "active"


def test_a_new_symptom_judged_serious_goes_to_a_doctor(client, graph):
    doctor(client)
    token = patient(client)
    conversation_id, episode = _due_episode(client, token, graph)
    _checkin(client, token, episode["id"], "better", new="now there is blood when I cough")

    graph.level = "level_2"
    turn = _turn(client, token, "It's getting better. Now there is blood when I cough",
                 conversation_id=conversation_id)

    assert turn["triage_level"] == "level_2"
    assert _sql("SELECT count(*) FROM review_cases").scalar_one() == 1
    assert _detail(client, token, conversation_id)["followup"]["state"] == "escalated"


def test_a_new_symptom_judged_self_care_is_not_raised_and_the_better_illness_closes(client, graph):
    token = patient(client)
    conversation_id, episode = _due_episode(client, token, graph)
    _checkin(client, token, episode["id"], "better", new="a blocked nose now")

    graph.level, graph.diagnosis = "level_1", "Common cold"
    turn = _turn(client, token, "It's getting better. A blocked nose now", conversation_id=conversation_id)

    assert turn["triage_level"] == "level_1"
    background = graph.inputs[-1]["patient_background"]
    assert "getting better" in background and "a blocked nose now" in background
    assert _sql("SELECT state FROM episodes WHERE id = :id", id=episode["id"]).scalar_one() == "resolved"
    assert turn["followup"]["id"] != episode["id"]  # the new symptom is watched in its own right


def test_the_same_with_something_new_keeps_the_schedule(client, graph):
    token = patient(client)
    conversation_id, episode = _due_episode(client, token, graph)

    r = _checkin(client, token, episode["id"], "same", new="a rash on my chest")
    assert r.json()["outcome"] == "reassess"

    graph.level = "level_1"
    turn = _turn(client, token, "It's about the same. A rash on my chest", conversation_id=conversation_id)

    assert turn["triage_level"] == "level_1"
    assert turn["followup"]["id"] == episode["id"] and turn["followup"]["next_checkin_at"] is not None
    assert _sql("SELECT awaiting_retriage FROM episodes WHERE id = :id", id=episode["id"]).scalar_one() is False


def test_a_sign_not_on_the_list_is_refused(client, graph):
    token = patient(client)
    _, episode = _due_episode(client, token, graph)
    assert _checkin(client, token, episode["id"], "worse", signs=["my left toe is purple"]).status_code == 422


def test_a_check_in_cannot_be_answered_before_it_is_due(client, graph):
    token = patient(client)
    _, episode = _episode(client, token, graph)
    assert _checkin(client, token, episode["id"], "better").status_code == 409


def test_another_patient_cannot_answer_or_change_an_episode(client, graph):
    alice = patient(client, email="alice@example.com")
    bob = patient(client, email="bob@example.com")
    _, episode = _due_episode(client, alice, graph)

    assert _checkin(client, bob, episode["id"], "better").status_code == 404
    assert client.post(f"/followups/{episode['id']}/stop", headers=auth(bob)).status_code == 404


def test_postponing_moves_the_open_check_in_later(client, graph):
    token = patient(client)
    conversation_id, episode = _due_episode(client, token, graph)

    r = client.post(f"/followups/{episode['id']}/postpone", json={"hours": 24}, headers=auth(token))

    assert r.status_code == 200
    followup_now = _detail(client, token, conversation_id)["followup"]
    assert followup_now["checkin_due"] is False
    assert datetime.fromisoformat(followup_now["next_checkin_at"]) > datetime.now(timezone.utc) + timedelta(hours=23)


def test_stopping_ends_check_ins_without_calling_it_recovery(client, graph):
    token = patient(client)
    conversation_id, episode = _episode(client, token, graph)

    client.post(f"/followups/{episode['id']}/stop", headers=auth(token))
    followup.run_due(now=_at(4))

    assert _detail(client, token, conversation_id)["followup"]["state"] == "stopped"
    assert outbox("checkin_due") == []


def test_a_patients_age_brings_in_the_child_signs(client, graph):
    token = patient(client)
    client.put("/me/profile", json={"date_of_birth": date(date.today().year - 3, 1, 1).isoformat(),
                                    "gender": None, "allergies": [], "medications": [], "conditions": []},
               headers=auth(token))

    _, episode = _episode(client, token, graph, diagnosis="Acute bronchitis")

    assert any("breastfeed" in s["text"].lower() for s in episode["return_signs"])
