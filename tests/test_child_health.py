"""Child-health mode: Q&A from IMNCI/WHO documents, with no triage. Two things are fixed in
code and tested here without a model: a WHO danger sign puts "go to hospital now" before
the answer, and every answer ends with the WHO "return immediately" signs."""
import json

import pytest

from app.graph.child_builder import build_child_graph
from app.graph.nodes import child_answer_node, child_retrieve_node, child_signs_node
from app.services import child_store
from app.services.child_health_content import (
    DANGER_SIGNS, GENERAL_LABEL, OVERRIDE_TEXT, UNCHECKED_TEXT, compose, counting_signs,
)
from app.services.followup_content import CHILD_BREATHING_SIGNS, CHILD_SIGNS

AGE_BANDS = ("under_2_months", "2_months_to_5_years", "over_5_years", "not_stated")
GENERAL = [k for k, s in DANGER_SIGNS.items() if not s["young_infant_only"]]
YOUNG_INFANT = [k for k, s in DANGER_SIGNS.items() if s["young_infant_only"]]


def _signs(*present, age_band="not_stated", checked=True):
    return {"checked": checked, "age_band": age_band, "present": list(present)}


def _answer(guidelines="Give extra fluids [1].", general="", cited=(1,), urgent=False):
    return {"from_guidelines": guidelines, "general": general, "possible_causes": [],
            "cited": list(cited), "urgent": urgent, "is_clarification": False}


PASSAGES = [{"id": "a", "source": "imci-chart-booklet.pdf", "page": 12, "text": "...", "score": 0.8},
            {"id": "b", "source": "imnci-chart.pdf", "page": 3, "text": "...", "score": 0.7}]


# ── which signs count ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("age_band", AGE_BANDS)
@pytest.mark.parametrize("sign", GENERAL)
def test_a_general_danger_sign_counts_at_any_age(sign, age_band):
    assert counting_signs([sign], age_band) == [sign]


@pytest.mark.parametrize("sign", YOUNG_INFANT)
def test_a_young_infant_sign_counts_only_under_two_months(sign):
    assert counting_signs([sign], "under_2_months") == [sign]
    for older in ("2_months_to_5_years", "over_5_years", "not_stated"):
        assert counting_signs([sign], older) == []


def test_the_lists_cover_what_was_agreed():
    assert {"cannot_drink", "vomits_everything", "convulsions", "lethargic"} <= set(GENERAL)
    assert "fever_or_cold" in YOUNG_INFANT


# ── what the parent sees ─────────────────────────────────────────────────────────────

def test_a_danger_sign_puts_go_to_hospital_before_the_answer():
    out = compose(_signs("convulsions"), _answer(), PASSAGES)

    assert out["response"].startswith(OVERRIDE_TEXT)
    assert out["response"].index(OVERRIDE_TEXT) < out["response"].index("Give extra fluids")
    assert out["child"]["override"] == {"reason": "signs", "signs": [DANGER_SIGNS["convulsions"]["text"]]}


def test_a_fever_in_a_baby_under_two_months_overrides_but_not_in_an_older_child():
    baby = compose(_signs("fever_or_cold", age_band="under_2_months"), _answer(), PASSAGES)
    toddler = compose(_signs("fever_or_cold", age_band="2_months_to_5_years"), _answer(), PASSAGES)

    assert baby["child"]["override"]["reason"] == "signs"
    assert toddler["child"]["override"] is None


def test_an_emergency_the_answer_flags_raises_the_banner_without_a_listed_sign():
    out = compose(_signs(), _answer(urgent=True), PASSAGES)

    assert out["response"].startswith(OVERRIDE_TEXT)
    assert out["child"]["override"] == {"reason": "urgent", "signs": []}


def test_when_the_sign_check_failed_the_danger_signs_lead_the_answer():
    out = compose(_signs(checked=False), _answer(), PASSAGES)

    assert out["response"].startswith(UNCHECKED_TEXT)
    assert out["child"]["override"]["reason"] == "unchecked"
    assert out["child"]["override"]["signs"] == [s["text"] for s in DANGER_SIGNS.values()]


def test_no_banner_when_nothing_is_wrong():
    out = compose(_signs(), _answer(), PASSAGES)

    assert out["child"]["override"] is None
    assert out["response"] == "Give extra fluids [1]."


def test_every_answer_ends_with_the_who_return_immediately_signs():
    out = compose(_signs(), _answer(), PASSAGES)

    assert out["child"]["watch_for"] == [s["text"] for s in CHILD_SIGNS + CHILD_BREATHING_SIGNS]


def test_only_passages_that_were_retrieved_can_be_cited():
    out = compose(_signs(), _answer(cited=(1, 7, "two", 1)), PASSAGES)

    assert out["child"]["sources"] == [{"n": 1, "source": "imci-chart-booklet.pdf", "page": 12}]


def test_general_knowledge_is_kept_apart_and_labelled():
    out = compose(_signs(), _answer(general="Most colds settle in a week."), PASSAGES)

    assert out["child"]["general"] == "Most colds settle in a week."
    assert out["child"]["guidelines"] == "Give extra fluids [1]."
    assert f"{GENERAL_LABEL} Most colds settle in a week." in out["response"]


def test_a_failed_answer_still_carries_the_banner_and_the_signs():
    out = compose(_signs("lethargic"), {}, PASSAGES)

    assert out["response"].startswith(OVERRIDE_TEXT)
    assert out["child"]["watch_for"]
    assert out["child"]["guidelines"] == "" and out["child"]["general"] == ""


# ── the nodes ────────────────────────────────────────────────────────────────────────

HISTORY = [{"role": "user", "content": "My son is 3 and has had loose stools since yesterday."},
           {"role": "assistant", "content": "Is he drinking normally?"}]


def _jev_answers(present=(), probability=0.9, age_band="2_months_to_5_years", age_probability=0.9):
    answers = {k: {"type": "choice", "probabilities": {"present": probability if k in present else 0.0,
                                                       "absent": 0.0, "not_mentioned": 1.0}}
               for k in DANGER_SIGNS}
    answers["age_band"] = {"type": "choice", "probabilities": {age_band: age_probability,
                                                               "not_stated": 1 - age_probability}}
    return answers


def test_retrieval_reads_the_question_with_the_parents_previous_message(monkeypatch):
    seen = []
    monkeypatch.setattr(child_store, "search", lambda query: seen.append(query) or PASSAGES)

    out = child_retrieve_node.child_retrieve_node({"user_input": "he is vomiting now", "chat_history": HISTORY})

    assert out == {"child_passages": PASSAGES}
    assert "loose stools" in seen[0] and "vomiting now" in seen[0]


def test_without_the_documents_the_answer_gets_no_passages_and_the_reason_is_kept(monkeypatch):
    def missing(query):
        raise child_store.ChildIndexMissing("no index")

    monkeypatch.setattr(child_store, "search", missing)

    out = child_retrieve_node.child_retrieve_node({"user_input": "fever", "chat_history": []})

    assert out == {"child_passages": [], "child_error": "no index"}


def test_the_sign_check_asks_the_age_and_every_sign_of_the_parents_words_in_one_call(jev):
    jev.answers = _jev_answers()

    child_signs_node.child_signs_node({"user_input": "he is vomiting now", "chat_history": HISTORY})

    call = jev.calls[-1]
    assert set(call["questions"]) == {"age_band", *DANGER_SIGNS}
    assert call["state"] == {"messages": [HISTORY[0]["content"], "he is vomiting now"]}


def test_a_sign_counts_from_a_low_probability_because_missing_one_is_the_harm(jev):
    jev.answers = _jev_answers(present=("convulsions",), probability=0.35)
    assert child_signs_node.child_signs_node({"user_input": "x", "chat_history": []})["child_signs"] == {
        "checked": True, "age_band": "2_months_to_5_years", "present": ["convulsions"]}

    jev.answers = _jev_answers(present=("convulsions",), probability=0.2)
    assert child_signs_node.child_signs_node({"user_input": "x", "chat_history": []})["child_signs"]["present"] == []


def test_the_age_band_is_used_only_when_jev_is_sure_of_it(jev):
    jev.answers = _jev_answers(age_band="under_2_months", age_probability=0.7)

    signs = child_signs_node.child_signs_node({"user_input": "x", "chat_history": []})["child_signs"]

    assert signs["age_band"] == "not_stated"


@pytest.mark.parametrize("answers", [None, {"age_band": {"probabilities": {"not_stated": 1.0}}}])
def test_an_unreachable_or_malformed_sign_check_is_marked_unchecked(jev, answers):
    jev.answers = answers  # None: unreachable; a dict missing the signs: malformed

    signs = child_signs_node.child_signs_node({"user_input": "x", "chat_history": []})["child_signs"]

    assert signs == {"checked": False, "age_band": "not_stated", "present": []}


def test_the_answer_prompt_numbers_the_passages_and_ends_with_the_question(monkeypatch):
    prompts = []
    reply = {"from_guidelines": "Give ORS [2].", "general": "", "possible_causes": ["gastroenteritis"],
             "cited": [2], "urgent": False, "is_clarification": False}
    monkeypatch.setattr(child_answer_node, "call_model", lambda p: prompts.append(p) or json.dumps(reply))

    out = child_answer_node.child_answer_node(
        {"user_input": "what should I give him?", "chat_history": HISTORY, "child_passages": PASSAGES})

    assert out == {"child_answer": reply}
    assert "[1] (imci-chart-booklet.pdf, page 12)" in prompts[0] and "[2] (imnci-chart.pdf, page 3)" in prompts[0]
    assert prompts[0].rstrip().endswith("what should I give him?")


def test_with_no_passages_the_model_is_told_so(monkeypatch):
    prompts = []
    monkeypatch.setattr(child_answer_node, "call_model", lambda p: prompts.append(p) or "{}")

    child_answer_node.child_answer_node({"user_input": "x", "chat_history": [], "child_passages": []})

    assert "No passages matched" in prompts[0]


def test_a_malformed_model_reply_leaves_an_empty_answer(monkeypatch):
    monkeypatch.setattr(child_answer_node, "call_model", lambda p: "sorry, no JSON here")

    out = child_answer_node.child_answer_node({"user_input": "x", "chat_history": [], "child_passages": []})

    assert out == {"child_answer": {}}


def test_the_child_graph_carries_every_child_key_to_its_result(monkeypatch, jev):
    """LangGraph drops keys its state schema doesn't declare, silently."""
    jev.answers = _jev_answers(present=("lethargic",))
    monkeypatch.setattr(child_store, "search", lambda query: PASSAGES)
    monkeypatch.setattr(child_answer_node, "call_model", lambda p: json.dumps(
        {"from_guidelines": "Keep him warm [1].", "cited": [1]}))

    result = build_child_graph().invoke({"conversation_id": "c1", "user_input": "he is floppy",
                                         "chat_history": []})

    assert result["child_signs"]["present"] == ["lethargic"]
    assert result["child_passages"] == PASSAGES
    assert result["child_answer"]["from_guidelines"] == "Keep him warm [1]."
