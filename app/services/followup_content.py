"""What follow-up tells a patient to watch for, and when it checks back.

Everything here is from published guidance, not written by a model, and there is no
clinician review behind it — a deliberate choice for this demo. Sources, read 2026-10-23:

- NICE NG84 sore throat, NG120 acute cough, NG79 sinusitis, NG91 otitis media, NG109 lower
  UTI: the usual course, when a person should seek help if things do not start to improve,
  and the signs that warrant hospital referral.
- WHO IMCI chart booklet (2014): the signs that mean a sick child must return immediately.

Signs are phrased as things a patient can notice, never as diagnoses. Each carries the
level it raises a case to: level_3 for "seek urgent care now", level_2 for "a doctor should
look at this".
"""
import re
from dataclasses import dataclass, field

# NICE's general wording, common to all five guidelines: seek medical help if symptoms
# "worsen rapidly or significantly ... or the person becomes systemically very unwell".
GENERAL_SIGNS = [
    {"text": "Your symptoms get worse quickly or a lot", "level": "level_2"},
    {"text": "You become very unwell in yourself", "level": "level_3"},
]

# WHO IMCI: "Advise mother to return immediately if the child has any of these signs".
CHILD_SIGNS = [
    {"text": "Your child is not able to drink or breastfeed", "level": "level_3"},
    {"text": "Your child becomes sicker", "level": "level_3"},
    {"text": "Your child develops a fever", "level": "level_3"},
]
# IMCI's extra signs for a child with a cough or cold.
CHILD_BREATHING_SIGNS = [
    {"text": "Your child is breathing fast", "level": "level_3"},
    {"text": "Your child has difficulty breathing", "level": "level_3"},
]


@dataclass(frozen=True)
class Complaint:
    label: str
    # The usual course as NICE tells patients; None where the guideline states none.
    usual_course: str | None
    # Days after the answer at which to check in. The last is NICE's "should have started to
    # improve by" point: still the same then means a doctor should look.
    checkin_days: tuple[int, ...]
    signs: list = field(default_factory=list)
    # Words that identify the complaint in a diagnosis or a patient's message.
    terms: tuple[str, ...] = ()
    child_breathing: bool = False


COMPLAINTS = {
    # NG84: "can last around 1 week"; back-up antibiotics "if symptoms do not start to
    # improve within 3 to 5 days"; seek help if they "do not start to improve after 1 week".
    "sore_throat": Complaint(
        label="Sore throat", usual_course="usually lasts about 1 week", checkin_days=(3, 7),
        terms=("sore throat", "pharyngitis", "tonsillitis", "strep throat"),
    ),
    # NG91: "about 3 days, can be up to 1 week"; seek help if it does "not start to improve
    # after 3 days"; review "if symptoms do not improve within 7 days". Referral for
    # mastoiditis or facial nerve paralysis.
    "ear_infection": Complaint(
        label="Ear infection", usual_course="usually lasts about 3 days, sometimes up to 1 week",
        checkin_days=(3, 7),
        signs=[{"text": "Swelling, redness or pain behind your ear", "level": "level_3"},
               {"text": "Weakness or drooping on one side of your face", "level": "level_3"}],
        terms=("otitis media", "ear infection", "earache", "ear ache", "ear pain"),
    ),
    # NG79: "usual course 2 to 3 weeks"; back-up antibiotics "if symptoms do not improve
    # within 7 days"; seek help if they "do not improve after 3 weeks". Referral for
    # periorbital swelling, double vision or reduced vision, swelling over the frontal bone,
    # severe frontal headache.
    "sinusitis": Complaint(
        label="Sinusitis", usual_course="usually lasts 2 to 3 weeks", checkin_days=(7, 21),
        signs=[{"text": "Swelling or redness around your eye", "level": "level_3"},
               {"text": "Double vision, or your eyesight gets worse", "level": "level_3"},
               {"text": "Swelling over your forehead", "level": "level_3"},
               {"text": "A severe headache at the front of your head", "level": "level_3"}],
        terms=("sinusitis", "rhinosinusitis", "sinus infection"),
    ),
    # NG120: "lasts up to 3 or 4 weeks"; seek help if symptoms "do not improve after 3 to 4
    # weeks". No earlier point is given, so there is one check-in, at the earlier bound.
    "cough": Complaint(
        label="Cough", usual_course="usually lasts up to 3 or 4 weeks", checkin_days=(21,),
        terms=("cough", "bronchitis", "chest infection"), child_breathing=True,
    ),
    # NG109: seek help if symptoms "do not start to improve within 48 hours"; fever is named
    # as a symptom of a kidney infection (pyelonephritis).
    "uti": Complaint(
        label="Urinary infection", usual_course="should start to improve within 48 hours",
        checkin_days=(2,),
        signs=[{"text": "You develop a fever", "level": "level_2"}],
        terms=("urinary tract infection", "uti", "cystitis", "bladder infection"),
    ),
    # No guideline fetched. My call: NG84's pattern (day 3, then day 7), NICE's commonest
    # "does not start to improve after 1 week".
    "general": Complaint(
        label="Your symptoms", usual_course=None, checkin_days=(3, 7), child_breathing=True,
    ),
}

# How a check-in's overall answer is put in words, in the transcript and to the agents.
OVERALL_WORDS = {"better": "getting better", "same": "about the same", "worse": "getting worse"}

# Checked in this order, so a specific complaint wins over a vaguer one ("UTI with a cough").
_ORDER = ("uti", "ear_infection", "sinusitis", "sore_throat", "cough")


def classify(*texts: str) -> str:
    """The complaint the texts describe, checked in the order given: pass the diagnosis
    first, so it outranks the patient's own words. 'general' when nothing matches."""
    for raw in texts:
        words = (raw or "").lower()
        for key in _ORDER:
            if any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", words) for t in COMPLAINTS[key].terms):
                return key
    return "general"


@dataclass(frozen=True)
class Plan:
    complaint: str
    label: str
    usual_course: str | None
    checkin_days: tuple[int, ...]
    return_signs: list


def plan_for(complaint: str, age: int | None = None) -> Plan:
    """The signs to watch and the check-in days for a complaint, for a patient of `age`."""
    c = COMPLAINTS.get(complaint, COMPLAINTS["general"])
    signs = list(GENERAL_SIGNS) + list(c.signs)
    if age is not None and age < 5:
        signs += CHILD_SIGNS + (CHILD_BREATHING_SIGNS if c.child_breathing else [])
    return Plan(complaint=complaint if complaint in COMPLAINTS else "general", label=c.label,
                usual_course=c.usual_course, checkin_days=c.checkin_days, return_signs=signs)
