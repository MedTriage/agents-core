"""Child-health turns through the API: a conversation's mode, what reaches the child graph,
and what is saved. Child mode does no triage, so none of the triage machinery may fire."""
import json

from sqlalchemy import text

from app.db.session import engine
from app.services.child_health_content import OVERRIDE_TEXT, WATCH_FOR
from tests.conftest import auth, patient


def _sql(query, **params):
    with engine.begin() as conn:
        return conn.execute(text(query), params)


def _ask(client, token, text="He has had loose stools since yesterday", conversation_id=None, mode="child"):
    body = {"text": text, "mode": mode, **({"conversation_id": conversation_id} if conversation_id else {})}
    r = client.post("/process", json=body, headers=auth(token))
    assert r.status_code == 200, r.text
    return r.json()


def _detail(client, token, conversation_id):
    return client.get(f"/conversations/{conversation_id}", headers=auth(token)).json()


def test_a_child_question_is_answered_from_the_guidelines_without_a_triage_level(client, graph, child_graph):
    token = patient(client)

    out = _ask(client, token)

    assert out["mode"] == "child" and out["triage_level"] is None
    assert out["response"] == "Give extra fluids [1]."
    assert out["child"]["sources"] == [{"n": 1, "source": "imci-chart-booklet.pdf", "page": 12}]
    assert out["child"]["watch_for"] == WATCH_FOR
    assert graph.inputs == [] and len(child_graph.inputs) == 1


def test_the_conversation_and_its_messages_are_saved_as_child_mode(client):
    token = patient(client)
    out = _ask(client, token)

    detail = _detail(client, token, out["conversation_id"])
    listed = client.get("/conversations", headers=auth(token)).json()

    assert detail["mode"] == "child" and listed[0]["mode"] == "child"
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["triage_level"] is None
    assert detail["messages"][1]["child"]["watch_for"] == WATCH_FOR
    assert detail["title"] == "He has had loose stools since yesterday"


def test_the_parents_profile_never_reaches_the_child_graph(client, child_graph):
    token = patient(client)
    client.put("/me/profile", json={"date_of_birth": "1990-02-03", "gender": "female", "allergies": ["penicillin"],
                                    "medications": [], "conditions": []}, headers=auth(token))

    _ask(client, token)

    sent = child_graph.inputs[-1]
    assert "patient_background" not in sent and "scribe_output" not in sent
    assert "penicillin" not in json.dumps(sent) and "1990" not in json.dumps(sent)


def test_a_danger_sign_puts_go_to_hospital_first_and_never_locks_the_conversation(client, child_graph):
    token = patient(client)
    child_graph.signs = {"checked": True, "age_band": "2_months_to_5_years", "present": ["convulsions"]}

    out = _ask(client, token, "He had a fit this morning")

    assert out["response"].startswith(OVERRIDE_TEXT)
    assert out["child"]["override"]["reason"] == "signs"
    assert _detail(client, token, out["conversation_id"])["locked"] is False
    _ask(client, token, "What should I watch for tonight?", conversation_id=out["conversation_id"])


def test_child_mode_opens_no_review_case_episode_or_profile_suggestion(client, child_graph):
    token = patient(client)
    child_graph.signs = {"checked": True, "age_band": "under_2_months", "present": ["fever_or_cold"]}

    _ask(client, token, "My 5 week old feels hot")

    for table in ("review_cases", "episodes", "record_items"):
        assert _sql(f"SELECT count(*) FROM {table}").scalar_one() == 0, table


def test_an_existing_conversation_keeps_its_mode_whatever_the_client_sends(client, graph, child_graph):
    token = patient(client)
    first = _ask(client, token)

    second = _ask(client, token, "And his temperature?", conversation_id=first["conversation_id"], mode="triage")

    assert second["mode"] == "child" and graph.inputs == [] and len(child_graph.inputs) == 2
    assert child_graph.inputs[-1]["chat_history"][0]["content"] == "He has had loose stools since yesterday"


def test_without_a_mode_a_conversation_is_triage_as_before(client, graph, child_graph):
    token = patient(client)

    r = client.post("/process", json={"text": "I have a sore throat"}, headers=auth(token))

    assert r.json()["triage_level"] == "level_1" and child_graph.inputs == [] and len(graph.inputs) == 1
    assert client.get("/conversations", headers=auth(token)).json()[0]["mode"] == "triage"


def test_an_unknown_mode_is_refused(client):
    token = patient(client)

    r = client.post("/process", json={"text": "hi", "mode": "drugs"}, headers=auth(token))

    assert r.status_code == 422


def test_a_streamed_child_turn_ends_with_the_child_answer(client):
    token = patient(client)

    r = client.post("/process/stream", json={"text": "He has a cough", "mode": "child"}, headers=auth(token))

    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    done = events[-1]
    assert done["type"] == "done" and done["mode"] == "child" and done["triage_level"] is None
    assert done["child"]["watch_for"] == WATCH_FOR
    assert {"type": "stage", "stage": "reading"} in events
