"""Child mode: which WHO danger signs the parent describes, and how old the child is.

One Jev call asks the age band and one question per sign, answered in parallel. Only the
parent's own messages are read, never the assistant's replies. What the signs lead to is
decided in code (child_health_content), not here. If Jev can't be reached or answers
malformed, the signs are marked unchecked, which puts the whole list in front of the parent.
"""
from app.config import CHILD_AGE_MIN_PROBABILITY, CHILD_SIGN_MIN_PROBABILITY
from app.services import jev
from app.services.child_health_content import DANGER_SIGNS

_AGE_CRITERIA = {
    "under_2_months": "The child is younger than 2 months: a newborn, or a baby of a few weeks",
    "2_months_to_5_years": "The child is between 2 months and 5 years old",
    "over_5_years": "The child is older than 5 years",
    "not_stated": "The child's age is not stated",
}
_SIGN_CRITERIA = {
    "present": "The parent says the child has this now, or has had it during this illness",
    "absent": "The parent says the child does not have this",
    "not_mentioned": "The parent does not mention this",
}
_UNCHECKED = {"checked": False, "age_band": "not_stated", "present": []}


def _questions() -> dict:
    questions = {"age_band": {"type": "choice", "criteria": _AGE_CRITERIA,
                              "instructions": "How old is the child the parent is asking about?"}}
    for key, sign in DANGER_SIGNS.items():
        questions[key] = {"type": "choice", "criteria": _SIGN_CRITERIA,
                          "instructions": f"Does the parent describe this in their child: {sign['text']}?"}
    return questions


def child_signs_node(state: dict) -> dict:
    words = [m["content"] for m in state.get("chat_history") or [] if m.get("role") == "user"]
    words.append(state["user_input"])
    try:
        answers = jev.decide({"messages": words}, _questions())
        # Recall first: a missed sign is the error that harms, so a low bar counts it.
        present = [k for k in DANGER_SIGNS
                   if answers[k]["probabilities"].get("present", 0) >= CHILD_SIGN_MIN_PROBABILITY]
        ages = answers["age_band"]["probabilities"]
        band = max(ages, key=ages.get)
    except Exception as e:
        print(e)
        return {"child_signs": dict(_UNCHECKED)}
    sure = ages[band] >= CHILD_AGE_MIN_PROBABILITY and band in _AGE_CRITERIA
    return {"child_signs": {"checked": True, "age_band": band if sure else "not_stated", "present": present}}
