"""The Jev client: one choice question to the pinned model, probabilities back. The network
is replaced at httpx; the request shape follows OpenRouter's decisions documentation."""
import httpx

from app.services.jev import choose  # the real function, bound before the conftest fake


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


def test_an_error_from_jev_is_raised_not_hidden(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(
        503, json={"error": "unavailable"}, request=httpx.Request("POST", url)))
    try:
        choose({}, "Which?", {"general": "None"})
    except httpx.HTTPStatusError:
        return
    raise AssertionError("a failed call must raise, so the caller falls back to the word list")
