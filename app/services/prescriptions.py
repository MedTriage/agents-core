"""What happens to a prescription after the prescriber drafts it.

A draft belongs to the reviewing doctor, who edits it and issues it under their own name. Reports
of India's 2020 Telemedicine Practice Guidelines say the final prescription must come from a
registered doctor, and this is the one place that decides what a patient can see, so a draft
stays out of every patient route unless the test switch below is on.
"""
from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import PRESCRIPTION_DEMO_INSTANT
from app.db.models import Conversation, Message, Prescription, ReviewCase, User
from app.graph.nodes.prescriber_node import allergy_conflicts
from app.services.patient_context import build_background
from app.services.review import case_question

_Required = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
_Item = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class Medication(BaseModel):
    drug: _Required
    dose: _Required
    frequency: _Required
    duration: _Required
    route: _Required
    notes: str = Field(default="", max_length=1000)


class FinalPrescription(BaseModel):
    """What a doctor may issue. An empty medication list means no medicine is to be issued."""
    diagnosis: str = Field(default="", max_length=500)
    medications: list[Medication] = Field(default_factory=list, max_length=20)
    refer_now: bool = False
    investigations: list[_Item] = Field(default_factory=list, max_length=30)
    advice: list[_Item] = Field(default_factory=list, max_length=30)
    return_precautions: list[_Item] = Field(default_factory=list, max_length=30)
    assumptions: list[_Item] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def _needs_a_diagnosis(self):
        if self.medications and not self.diagnosis.strip():
            raise ValueError("A prescription needs a diagnosis")
        return self


class NotIssuable(Exception):
    """The draft is incomplete, so the doctor has to edit it first."""


class AllergyConflict(Exception):
    def __init__(self, conflicts: list[dict]):
        super().__init__("allergy conflict")
        self.conflicts = conflicts


def _visible_statuses() -> tuple[str, ...]:
    return ("issued", "draft") if PRESCRIPTION_DEMO_INSTANT else ("issued",)


def _content(rx: Prescription) -> dict:
    return rx.final or rx.draft or {}


def _visible(rx: Prescription) -> bool:
    return rx.status in _visible_statuses() and bool(_content(rx))


def hospital_flag(rx: Prescription | None) -> bool:
    """The prescriber said "go to hospital now" on a case the guardian held: the doctor sees it
    first. It moves no level; the guardian decided that."""
    return bool(rx and rx.status == "draft" and (rx.draft or {}).get("refer_now") is True)


def save_draft(db: Session, case: ReviewCase, result: dict) -> Prescription | None:
    """Keep what the prescriber came back with, if it ran. The caller commits."""
    draft, problems = result.get("prescription"), result.get("prescription_problems") or []
    if draft is None and not problems:
        return None
    rx = Prescription(case_id=case.id, message_id=case.message_id, conversation_id=case.conversation_id,
                      patient_id=case.patient_id, status="draft", draft=draft, problems=problems,
                      attempts=result.get("prescription_attempts") or 0)
    db.add(rx)
    db.flush()
    return rx


def for_case(db: Session, case_id) -> Prescription | None:
    return db.scalar(select(Prescription).where(Prescription.case_id == case_id))


def for_cases(db: Session, case_ids) -> dict:
    return {rx.case_id: rx for rx in db.scalars(select(Prescription).where(Prescription.case_id.in_(case_ids)))}


def _allergy_state(db: Session, case: ReviewCase) -> dict:
    """Everything known about the patient's allergies: what the agents saw, and the profile as
    it stands now, since the patient may have added one while the case waited."""
    conversation = db.get(Conversation, case.conversation_id)
    known = [(case.context_snapshot or {}).get("background"),
             build_background(db, case.patient_id, case.conversation_id)]
    return {"scribe_output": conversation.scribe_record,
            "patient_background": "\n".join(text for text in known if text),
            "user_input": case_question(db, case)}


def decide(db: Session, case: ReviewCase, doctor: User, approved: bool, edited: FinalPrescription | None,
           confirm_allergy: bool = False) -> None:
    """Apply the doctor's decision to the case's prescription. Raises NotIssuable or
    AllergyConflict before changing anything. The caller commits."""
    rx = for_case(db, case.id)
    if not approved:
        if rx is not None:
            rx.status = "discarded"
        return

    final = edited
    if final is None and rx is not None and rx.draft is not None:
        try:
            final = FinalPrescription.model_validate(rx.draft)
        except ValidationError as invalid:
            raise NotIssuable("This draft is incomplete. Edit it before approving.") from invalid
    if final is None or not final.medications:
        if rx is not None:
            rx.status = "discarded"
        return

    content = final.model_dump()
    if not confirm_allergy:
        conflicts = allergy_conflicts(content["medications"], _allergy_state(db, case))
        if conflicts:
            raise AllergyConflict([{"drug": drug, "allergy": allergy} for drug, allergy in conflicts])

    if rx is None:
        rx = Prescription(case_id=case.id, message_id=case.message_id, conversation_id=case.conversation_id,
                          patient_id=case.patient_id)
        db.add(rx)
    rx.status, rx.final = "issued", content
    rx.doctor_id, rx.issued_by_name, rx.issued_by_registration = doctor.id, doctor.name, doctor.registration_number
    rx.issued_at = datetime.now(timezone.utc)


# ── what each side sees ──────────────────────────────────────────────────────────────

def doctor_view(rx: Prescription | None) -> dict | None:
    if rx is None:
        return None
    return {"id": str(rx.id), "status": rx.status, "draft": rx.draft, "problems": rx.problems or [],
            "final": rx.final, "hospital_flag": hospital_flag(rx),
            "issued_by_name": rx.issued_by_name, "issued_by_registration": rx.issued_by_registration,
            "issued_at": rx.issued_at.isoformat() if rx.issued_at else None}


def link_out(rx: Prescription | None) -> dict | None:
    """Where to find it, and nothing of what it says. None when the patient may not see it."""
    if rx is None or not _visible(rx):
        return None
    return {"id": str(rx.id), "status": rx.status, "message_id": rx.message_id}


def summary_out(rx: Prescription) -> dict:
    content = _content(rx)
    return {
        **link_out(rx),
        "conversation_id": str(rx.conversation_id),
        "diagnosis": content.get("diagnosis"),
        "medications": content.get("medications") or [],
        "refer_now": bool(content.get("refer_now")),
        "issued_by_name": rx.issued_by_name,
        "issued_by_registration": rx.issued_by_registration,
        "issued_at": rx.issued_at.isoformat() if rx.issued_at else None,
        "created_at": rx.created_at.isoformat(),
    }


def sheet_out(db: Session, rx: Prescription) -> dict:
    """The printable sheet. A test draft leaves out the answer, which is still held for the doctor."""
    content = _content(rx)
    case = db.get(ReviewCase, rx.case_id)
    answer = db.get(Message, rx.message_id).content if rx.status == "issued" else None
    return {
        **summary_out(rx),
        "patient_message": case_question(db, case),
        "answer": answer,
        "investigations": content.get("investigations") or [],
        "advice": content.get("advice") or [],
        "return_precautions": content.get("return_precautions") or [],
        "assumptions": content.get("assumptions") or [],
    }


def of_patient(db: Session, user_id) -> list[Prescription]:
    rows = db.scalars(select(Prescription).where(Prescription.patient_id == user_id)
                      .order_by(Prescription.created_at.desc()))
    return [rx for rx in rows if _visible(rx)]


def own_visible(db: Session, user_id, prescription_id) -> Prescription | None:
    """Someone else's prescription, or a draft, is indistinguishable from a missing one."""
    rx = db.get(Prescription, prescription_id)
    return rx if rx is not None and rx.patient_id == user_id and _visible(rx) else None


def links_for_conversation(db: Session, conversation_id) -> list[dict]:
    rows = db.scalars(select(Prescription).where(Prescription.conversation_id == conversation_id)
                      .order_by(Prescription.created_at))
    return [link_out(rx) for rx in rows if _visible(rx)]


def issued_for_export(db: Session, user_id) -> list[dict]:
    """Issued only, even with the test switch on: a draft is not the patient's record."""
    rows = db.scalars(select(Prescription).where(Prescription.patient_id == user_id,
                                                 Prescription.status == "issued")
                      .order_by(Prescription.created_at))
    return [sheet_out(db, rx) for rx in rows]
