"""What child-health mode fixes in code: the WHO danger signs that put "go to hospital now"
before any answer, and the "return immediately" signs every answer ends with.

Child mode does no triage. It answers questions from IMNCI/WHO documents, and these two
pieces are the only rules: a sign the parent describes can only add urgency, never remove it.

Sources: WHO IMCI chart booklet (2014), "check for general danger signs" (2 months to 5
years) and "check for very severe disease" (stiff neck, stridor in a calm child); the WHO
young-infant chart (birth to 2 months). Rephrased as things a parent can notice. Before
real use, check them against the current MoHFW IMNCI edition, and check each document's
licence; neither has been done.
"""
from app.services.followup_content import CHILD_BREATHING_SIGNS, CHILD_SIGNS

DANGER_SIGNS = {
    "cannot_drink": {"text": "Not able to drink or breastfeed", "young_infant_only": False},
    "vomits_everything": {"text": "Vomits everything", "young_infant_only": False},
    "convulsions": {"text": "Has had a fit (convulsion), or is having one now", "young_infant_only": False},
    "lethargic": {"text": "Unusually sleepy, floppy or hard to wake, or unconscious",
                  "young_infant_only": False},
    "stiff_neck": {"text": "A stiff neck, with a fever", "young_infant_only": False},
    "stridor": {"text": "A harsh noise when breathing in, while calm", "young_infant_only": False},
    # Young infants (birth to 2 months): any of these is very severe disease in the WHO chart.
    "not_feeding_well": {"text": "Not feeding well", "young_infant_only": True},
    "fever_or_cold": {"text": "A fever, or feels cold to the touch", "young_infant_only": True},
    "breathing": {"text": "Breathing fast or with difficulty, or the chest pulling in",
                  "young_infant_only": True},
    "little_movement": {"text": "Moves only when touched or picked up, or not at all",
                        "young_infant_only": True},
}

OVERRIDE_TEXT = ("Go to the nearest hospital or emergency department now. What you describe can be "
                 "a sign of serious illness in a child.")
UNCHECKED_TEXT = "If your child has any of these signs, go to the nearest hospital now:"
GENERAL_LABEL = "General information, not from the WHO/IMNCI guidelines:"
FAILED_TEXT = "I couldn't answer that just now. Please try again in a moment."

# The WHO "return immediately" list follow-up already shows for under-5s, on every answer.
WATCH_FOR = [s["text"] for s in CHILD_SIGNS + CHILD_BREATHING_SIGNS]


def counting_signs(present: list[str], age_band: str) -> list[str]:
    """The reported signs that call for hospital now at this age. A young-infant sign counts
    only when the child is known to be under 2 months; the general ones count at any age."""
    return [k for k in present if k in DANGER_SIGNS
            and (not DANGER_SIGNS[k]["young_infant_only"] or age_band == "under_2_months")]


def _override(signs: dict, answer: dict) -> dict | None:
    counted = counting_signs(signs.get("present") or [], signs.get("age_band") or "not_stated")
    if counted:
        return {"reason": "signs", "signs": [DANGER_SIGNS[k]["text"] for k in counted]}
    if answer.get("urgent"):
        return {"reason": "urgent", "signs": []}
    if not signs.get("checked"):
        # Nobody could check for the signs, so the parent checks: the whole list leads.
        return {"reason": "unchecked", "signs": [s["text"] for s in DANGER_SIGNS.values()]}
    return None


def compose(signs: dict, answer: dict, passages: list[dict]) -> dict:
    """What the parent sees: any banner first, then the answer from the guidelines, then
    general knowledge kept apart and labelled. `response` is the stored message text; `child`
    is what the client renders, the footer and cited sources included."""
    override = _override(signs, answer)
    guidelines = (answer.get("from_guidelines") or "").strip()
    general = (answer.get("general") or "").strip()

    parts = []
    if override and override["reason"] == "unchecked":
        parts.append(UNCHECKED_TEXT + "\n" + "\n".join(f"- {s}" for s in override["signs"]))
    elif override:
        parts.append(OVERRIDE_TEXT)
    if guidelines:
        parts.append(guidelines)
    if general:
        parts.append(f"{GENERAL_LABEL} {general}")
    if not (guidelines or general):
        parts.append(FAILED_TEXT)

    # Only a passage that was retrieved can be cited; anything else the model names is dropped.
    cited = {n for n in answer.get("cited") or [] if isinstance(n, int)}
    sources = [{"n": n, "source": p["source"], "page": p["page"]}
               for n, p in enumerate(passages, start=1) if n in cited]

    return {
        "response": "\n\n".join(parts),
        "child": {
            "override": override,
            "guidelines": guidelines,
            "general": general,
            "possible_causes": list(answer.get("possible_causes") or []),
            "sources": sources,
            "watch_for": WATCH_FOR,
            "clarification": bool(answer.get("is_clarification")),
        },
    }
