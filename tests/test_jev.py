"""The Jev client: one choice question to the pinned model, probabilities back. The network
is replaced at httpx; the request shape follows OpenRouter's decisions documentation."""
import httpx

from app.services.jev import choose, decide  # the real functions, bound before the conftest fake


def test_one_choice_question_goes_to_the_pinned_model_and_its_probabilities_come_back(monkeypatch):
    sent = {}

    def post(url, json, headers, timeout):
        sent.update(url=url, json=json, headers=headers)
        body = {"answers": {"complaint": {"type": "choice", "choice": "uti", "confidence": 0.8,
                                          "probabilities": {"uti": 0.9, "general": 0.1}}}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)

    probabilities = choose({"diagnosis": "Cystitis"}, "Which complaint?",
                           {"uti": "Urinary infection", "general": "None of these"})

    assert probabilities == {"uti": 0.9, "general": 0.1}
    assert sent["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert sent["json"]["model"] == "typesafe/jev-1.13"
    assert sent["json"]["state"] == {"diagnosis": "Cystitis"}
    assert sent["json"]["questions"] == {"complaint": {
        "type": "choice", "instructions": "Which complaint?",
        "criteria": {"uti": "Urinary infection", "general": "None of these"}}}
    assert sent["headers"]["Authorization"].startswith("Bearer ")


def test_several_questions_go_in_one_call_and_every_answer_comes_back(monkeypatch):
    sent = {}
    answers = {"age_band": {"type": "choice", "choice": "under_2_months",
                            "probabilities": {"under_2_months": 0.9, "not_stated": 0.1}},
               "convulsions": {"type": "choice", "choice": "absent",
                               "probabilities": {"present": 0.05, "absent": 0.9, "not_mentioned": 0.05}}}

    def post(url, json, headers, timeout):
        sent.update(json=json)
        return httpx.Response(200, json={"answers": answers}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    questions = {"age_band": {"type": "choice", "instructions": "How old?", "criteria": {"under_2_months": "<2m"}},
                 "convulsions": {"type": "choice", "instructions": "Fits?", "criteria": {"present": "yes"}}}

    assert decide({"messages": ["my baby"]}, questions) == answers
    assert sent["json"] == {"model": "typesafe/jev-1.13", "state": {"messages": ["my baby"]},
                            "questions": questions}


def test_an_error_from_jev_is_raised_not_hidden(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(
        503, json={"error": "unavailable"}, request=httpx.Request("POST", url)))
    try:
        choose({}, "Which?", {"general": "None"})
    except httpx.HTTPStatusError:
        return
    raise AssertionError("a failed call must raise, so the caller falls back to the word list")
