"""The prescriber node: a draft prescription on level_2 turns only, for a doctor to edit and
issue. The model is faked; what is tested is the validation, the allergy rules and the routing,
which are code."""
import json

import pytest

from app.graph import builder
from app.graph.nodes import prescriber_node as node

GOOD = {
    "diagnosis": "Viral pharyngitis",
    "medications": [{"drug": "paracetamol", "dose": "1 g", "frequency": "every 6 hours",
                     "duration": "3 days", "route": "oral", "notes": ""}],
    "refer_now": False,
    "investigations": [],
    "advice": ["Rest and drink fluids."],
    "return_precautions": ["Trouble breathing."],
    "assumptions": [],
}


def _rx(drug="paracetamol", **changes):
    return {**GOOD, "medications": [{**GOOD["medications"][0], "drug": drug}], **changes}


def _state(**extra):
    return {"user_input": "My throat hurts", "scribe_output": {"allergies": []},
            "patient_background": None, **extra}


# ── the design's validation ──────────────────────────────────────────────────────────

def test_a_complete_prescription_passes_validation():
    assert node.validate(GOOD, _state()) == []


@pytest.mark.parametrize("field", ["drug", "dose", "frequency", "duration", "route"])
@pytest.mark.parametrize("value", ["", "as directed", "unknown", None])
def test_a_medication_field_must_be_specific(field, value):
    med = {**GOOD["medications"][0], field: value}
    problems = node.validate({**GOOD, "medications": [med]}, _state())
    assert any(f"'{field}' must be specific" in p for p in problems)


def test_at_least_one_medication_is_required():
    assert any("at least one medication" in p for p in node.validate({**GOOD, "medications": []}, _state()))


def test_return_precautions_and_refer_now_are_required():
    problems = node.validate({**GOOD, "return_precautions": [], "refer_now": "no"}, _state())
    assert "return_precautions is empty" in problems
    assert "refer_now must be true or false" in problems


# ── allergies, checked in code ───────────────────────────────────────────────────────

def test_an_allergy_in_the_message_rules_out_its_class():
    state = _state(user_input="I'm allergic to penicillin and my throat hurts")
    assert any("contraindicated" in p for p in node.validate(_rx("co-amoxiclav"), state))


def test_a_drug_merely_mentioned_in_the_message_is_not_an_allergy():
    state = _state(user_input="I took a course of penicillin last year, it cleared my chest")
    assert node.validate(_rx("amoxicillin"), state) == []


def test_an_allergy_in_the_scribe_record_counts_without_the_word_allergy_nearby():
    """The design's scan looked for "allerg" within 80 characters even in the record, whose
    entries are bare names, so a recorded penicillin allergy did not block amoxicillin."""
    state = _state(scribe_output={"allergies": ["penicillin"]})
    assert any("contraindicated" in p for p in node.validate(_rx("amoxicillin"), state))


def test_an_allergy_in_the_stated_profile_rules_out_its_class():
    background = ("Stated by the patient in their account profile:\n- Age: 52\n- Gender: female\n"
                  "- Allergies: penicillin (rash)\n- Regular medications: none listed\n"
                  "- Long-term conditions: none listed")
    assert any("contraindicated" in p for p in node.validate(_rx("amoxicillin"), _state(patient_background=background)))


def test_no_stated_allergy_does_not_ban_the_regular_medicine_on_the_next_line():
    """A plain scan of the background finds "Allergies" within 80 characters of amoxicillin."""
    background = ("Stated by the patient in their account profile:\n- Age: 52\n- Gender: female\n"
                  "- Allergies: none listed\n- Regular medications: amoxicillin\n"
                  "- Long-term conditions: none listed")
    assert node.validate(_rx("amoxicillin"), _state(patient_background=background)) == []


def test_an_allergy_mentioned_in_an_earlier_chat_counts_though_unconfirmed():
    background = ("Mentioned in earlier conversations, not confirmed by the patient:\n"
                  "- Allergy: penicillin (last mentioned 2026-11-01)")
    assert any("contraindicated" in p for p in node.validate(_rx("amoxicillin"), _state(patient_background=background)))


def test_a_stated_allergy_outside_the_class_table_still_blocks_that_drug():
    state = _state(scribe_output={"allergies": ["cephalexin"]})
    assert any("contraindicated" in p for p in node.validate(_rx("Cephalexin"), state))


# ── the retry loop ───────────────────────────────────────────────────────────────────

def test_the_first_valid_answer_is_used(monkeypatch):
    calls = []
    monkeypatch.setattr(node, "call_model", lambda prompt: calls.append(prompt) or json.dumps(GOOD))

    result = node.prescribe(_state())

    assert result == {"prescription": GOOD, "attempts": 1, "problems": []}
    assert len(calls) == 1


def test_a_rejected_answer_goes_back_with_its_problems(monkeypatch):
    answers = iter([json.dumps(_rx("amoxicillin")), json.dumps(GOOD)])
    prompts = []
    monkeypatch.setattr(node, "call_model", lambda prompt: prompts.append(prompt) or next(answers))

    result = node.prescribe(_state(user_input="I'm allergic to penicillin"))

    assert result["attempts"] == 2 and result["problems"] == []
    assert "YOUR PREVIOUS PRESCRIPTION WAS REJECTED" in prompts[1]
    assert "amoxicillin is contraindicated" in prompts[1]


def test_after_every_attempt_fails_the_last_draft_comes_back_flagged(monkeypatch):
    calls = []
    monkeypatch.setattr(node, "call_model", lambda prompt: calls.append(1) or json.dumps(_rx("amoxicillin")))

    result = node.prescribe(_state(user_input="I'm allergic to penicillin"))

    assert len(calls) == node.MAX_ATTEMPTS
    assert result["prescription"]["medications"][0]["drug"] == "amoxicillin"
    assert result["problems"]


def test_nonsense_from_the_model_is_retried_and_never_raises(monkeypatch):
    monkeypatch.setattr(node, "call_model", lambda prompt: "I cannot help with that")

    result = node.prescribe(_state())

    assert result["prescription"] is None and result["problems"]


def test_an_unreachable_model_gives_up_at_once(monkeypatch):
    """call_model has already backed off; four more rounds would leave the patient waiting."""
    calls = []

    def down(prompt):
        calls.append(1)
        raise ConnectionError("model is down")

    monkeypatch.setattr(node, "call_model", down)

    result = node.prescribe(_state())

    assert len(calls) == 1
    assert result["prescription"] is None and "ConnectionError" in result["problems"][0]


# ── what the model is told ───────────────────────────────────────────────────────────

def test_the_model_sees_the_patient_background(monkeypatch):
    prompts = []
    monkeypatch.setattr(node, "call_model", lambda prompt: prompts.append(prompt) or json.dumps(GOOD))

    node.prescribe(_state(patient_background="- Latest weight: 72 kg (recorded 3 Nov 2026)"))

    assert "PATIENT BACKGROUND" in prompts[0] and "72 kg" in prompts[0]


def test_the_node_returns_only_the_keys_it_sets(monkeypatch):
    monkeypatch.setattr(node, "call_model", lambda prompt: json.dumps(GOOD))
    assert node.prescriber_node(_state()) == {
        "prescription": GOOD, "prescription_attempts": 1, "prescription_problems": []}


# ── where it sits in the graph ───────────────────────────────────────────────────────

@pytest.fixture
def real_graph(monkeypatch):
    """The real wiring and state schema, with every model-calling node stubbed."""
    ran = []

    def stub(name, update):
        def run(state):
            ran.append(name)
            return update
        monkeypatch.setattr(builder, name, run)

    stub("scribe_node", {"scribe_output": {"turn": 1}})
    stub("parametric_node", {"parametric_output": {}})
    stub("kgrag_node", {"kgrag_output": {}})
    stub("mcp_node", {"mcp_output": {}})
    stub("dst_node", {"dst_output": {}})
    stub("orchestrator_node", {"orchestrator_response": "Answer.", "branch_refinement_hints": {}})
    monkeypatch.setattr(builder, "prescriber_node", lambda state: (ran.append("prescriber_node"), {
        "prescription": GOOD, "prescription_attempts": 1, "prescription_problems": []})[1])

    def build(level, intent="clinical_query"):
        stub("intent_node", {"intent_type": intent})
        stub("guardian_node", {"guardian_output": {"triage_level": level}, "triage_level": level})
        return builder.build_graph(), ran

    return build


def _input():
    return {"conversation_id": "c1", "user_input": "My throat hurts", "chat_history": []}


def test_a_level_2_turn_reaches_the_prescriber_and_the_draft_survives_the_schema(real_graph):
    graph, ran = real_graph("level_2")
    result = graph.invoke(_input())
    assert ran[-1] == "prescriber_node"
    assert result["prescription"] == GOOD
    assert result["prescription_attempts"] == 1


@pytest.mark.parametrize("level", ["level_1", "level_3"])
def test_level_1_and_level_3_never_reach_the_prescriber(real_graph, level):
    graph, ran = real_graph(level)
    result = graph.invoke(_input())
    assert "prescriber_node" not in ran
    assert result.get("prescription") is None


def test_chitchat_never_reaches_the_prescriber_even_at_level_2(real_graph):
    graph, ran = real_graph("level_2", intent="chitchat")
    graph.invoke(_input())
    assert "prescriber_node" not in ran
