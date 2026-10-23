"""The graph's side of persistence: the scribe takes its prior record from the state the
server built (not from a file), and every agent that reads patient context also sees the
account background — on the first turn too."""
import json

from app.graph.nodes import kgrag_node, mcp_node, orchestrator_node, parametric_node, scribe_node

BACKGROUND = ("Stated by the patient in their account profile:\n- Age: 52\n- Gender: female\n"
              "- Allergies: penicillin\n- Regular medications: none listed\n"
              "- Long-term conditions: none listed")


def _first_turn_state(**extra):
    return {
        "conversation_id": "c1", "user_input": "I have a cough", "chat_history": [],
        "scribe_output": {"turn": 1, "symptoms": [{"text": "cough", "first_turn": 1, "last_turn": 1}],
                          "current_conditions": [], "past_conditions": [], "medications": [],
                          "allergies": [], "demographics": {}, "notes": []},
        "patient_background": BACKGROUND,
        **extra,
    }


# ── scribe ───────────────────────────────────────────────────────────────────────────

def test_the_scribe_builds_on_the_prior_record_it_is_given(monkeypatch):
    monkeypatch.setattr(scribe_node, "call_model", lambda prompt: json.dumps(
        {"symptoms": ["cough", "fever"], "reported_this_turn": ["fever"]}))
    prior = {"turn": 3, "allergies": ["latex"], "medications": [], "current_conditions": [],
             "past_conditions": [], "demographics": {}, "notes": [],
             "symptoms": [{"text": "cough", "first_turn": 1, "last_turn": 2}]}

    record = scribe_node.scribe_node({"conversation_id": "c1", "user_input": "now a fever too",
                                      "scribe_output": prior})["scribe_output"]

    assert record["turn"] == 4
    assert record["allergies"] == ["latex"]
    assert record["symptoms"] == [{"text": "cough", "first_turn": 1, "last_turn": 2},
                                  {"text": "fever", "first_turn": 4, "last_turn": 4}]


def test_the_scribe_starts_a_first_turn_from_an_empty_record(monkeypatch):
    seen = {}

    def model(prompt):
        seen["prompt"] = prompt
        return json.dumps({"symptoms": ["headache"], "reported_this_turn": ["headache"]})
    monkeypatch.setattr(scribe_node, "call_model", model)

    record = scribe_node.scribe_node({"conversation_id": "c1", "user_input": "headache",
                                      "scribe_output": None})["scribe_output"]

    assert record["turn"] == 1
    assert record["symptoms"] == [{"text": "headache", "first_turn": 1, "last_turn": 1}]
    assert '"allergies": []' in seen["prompt"]


def test_a_failed_scribe_turn_keeps_the_prior_record_and_does_not_advance(monkeypatch):
    def broken(prompt):
        raise RuntimeError("model down")
    monkeypatch.setattr(scribe_node, "call_model", broken)
    prior = {"turn": 2, "allergies": ["latex"], "medications": [], "current_conditions": [],
             "past_conditions": [], "demographics": {}, "notes": [], "symptoms": []}

    record = scribe_node.scribe_node({"conversation_id": "c1", "user_input": "hi",
                                      "scribe_output": prior})["scribe_output"]

    assert record["turn"] == 2
    assert record["allergies"] == ["latex"]
    assert "error" in record


# ── every agent sees the background ──────────────────────────────────────────────────

def test_the_parametric_branch_sees_the_background_on_a_first_turn():
    assert BACKGROUND in parametric_node._build_context(_first_turn_state())


def test_the_kgrag_branch_sees_the_background_on_a_first_turn():
    assert BACKGROUND in kgrag_node._build_query_context(_first_turn_state())


def test_the_mcp_branch_sees_the_background():
    assert BACKGROUND in mcp_node._build_query_context(_first_turn_state())


def test_no_background_section_when_nothing_is_known():
    state = _first_turn_state(patient_background=None)
    for context in (parametric_node._build_context(state), kgrag_node._build_query_context(state),
                    mcp_node._build_query_context(state)):
        assert "PATIENT BACKGROUND" not in context


_SYNTHESIS = json.dumps({
    "response": "Likely a viral cough.", "is_supported": True, "clinical_issues": [],
    "safety_risk": "low", "decision": "approve", "confidence_adjusted": 0.8,
    "is_emergency": False, "is_clarification": False, "probable_diagnosis": "Viral cough",
    "differentials": [], "evidence_grounded": False, "conflicts": [], "issues": [], "branch_issues": {},
})


def _orchestrate(monkeypatch, parametric_output):
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return _SYNTHESIS
    monkeypatch.setattr(orchestrator_node, "call_model", model)
    orchestrator_node.orchestrator_node(_first_turn_state(
        intent_type="clinical_query", parametric_output=parametric_output,
        kgrag_output=None, mcp_output=None))
    return prompts


def test_the_orchestrator_sees_the_background_when_synthesising(monkeypatch):
    prompts = _orchestrate(monkeypatch, {
        "probable_diagnosis": "Viral cough", "differentials": [], "recommended_actions": [],
        "citations": [], "confidence": 0.8})
    assert len(prompts) == 1
    assert BACKGROUND in prompts[0]


def test_the_orchestrator_sees_the_background_when_answering_without_evidence(monkeypatch):
    prompts = _orchestrate(monkeypatch, {
        "probable_diagnosis": "Insufficient evidence", "differentials": [],
        "recommended_actions": [], "citations": [], "confidence": 0.0})
    assert len(prompts) == 1
    assert BACKGROUND in prompts[0]


# ── the real graph carries the server's inputs to the nodes ──────────────────────────

def test_the_compiled_graph_delivers_the_background_and_prior_record_to_its_nodes(monkeypatch):
    from app.graph import builder

    seen = {}

    def recorder(name, update):
        def node(state):
            seen[name] = dict(state)
            return update
        return node

    monkeypatch.setattr(builder, "scribe_node", recorder("scribe", {"scribe_output": {"turn": 2}}))
    monkeypatch.setattr(builder, "intent_node", recorder("intent", {"intent_type": "chitchat"}))
    monkeypatch.setattr(builder, "orchestrator_node", recorder("orchestrator", {
        "orchestrator_response": "hello", "branch_refinement_hints": {}}))
    monkeypatch.setattr(builder, "guardian_node", recorder("guardian", {"triage_level": "level_1"}))

    builder.build_graph().invoke({
        "conversation_id": "c1", "user_input": "hi", "chat_history": [],
        "scribe_output": {"turn": 1}, "patient_background": BACKGROUND,
    })

    assert seen["scribe"]["scribe_output"] == {"turn": 1}
    assert seen["scribe"]["patient_background"] == BACKGROUND
    assert seen["orchestrator"]["patient_background"] == BACKGROUND
