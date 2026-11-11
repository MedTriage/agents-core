"""The patient dashboard's data: their readings, their follow-up calendar, their prescriptions.
Every route is the caller's own data, so someone else's row is a 404."""
import uuid
from datetime import date, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import require_patient
from app.db.models import Episode, HealthReading, User
from app.db.session import get_db
from app.services import followup, prescriptions, readings

router = APIRouter(prefix="/me", tags=["dashboard"])


class ReadingIn(BaseModel):
    kind: Literal["weight", "height", "pulse", "blood_pressure"]
    value: float
    value2: float | None = None
    measured_on: date | None = None

    @model_validator(mode="after")
    def _plausible(self):
        self.value, self.value2 = readings.check(self.kind, self.value, self.value2)
        # One day's grace: a patient in India is often already on tomorrow by the server's clock.
        if self.measured_on and self.measured_on > date.today() + timedelta(days=1):
            raise ValueError("The date can't be in the future")
        return self


@router.get("/readings")
def list_readings(user: User = Depends(require_patient), db: Session = Depends(get_db)):
    return [readings.reading_out(r) for r in readings.of_patient(db, user.id)]


@router.post("/readings", status_code=201)
def add_reading(body: ReadingIn, user: User = Depends(require_patient), db: Session = Depends(get_db)):
    reading = HealthReading(user_id=user.id, kind=body.kind, value=body.value, value2=body.value2,
                            measured_on=body.measured_on or date.today())
    db.add(reading)
    db.commit()
    return readings.reading_out(reading)


@router.delete("/readings/{reading_id}", status_code=204)
def delete_reading(reading_id: int, user: User = Depends(require_patient), db: Session = Depends(get_db)):
    reading = db.get(HealthReading, reading_id)
    if reading is None or reading.user_id != user.id:
        raise HTTPException(404, "No such reading.")
    db.delete(reading)
    db.commit()
    return Response(status_code=204)


@router.get("/followups")
def list_followups(user: User = Depends(require_patient), db: Session = Depends(get_db)):
    episodes = db.scalars(select(Episode).where(Episode.user_id == user.id)
                          .order_by(Episode.opened_at.desc())).all()
    return [followup.calendar_out(db, episode) for episode in episodes]


@router.get("/prescriptions")
def list_prescriptions(user: User = Depends(require_patient), db: Session = Depends(get_db)):
    return [prescriptions.summary_out(rx) for rx in prescriptions.of_patient(db, user.id)]


@router.get("/prescriptions/{prescription_id}")
def get_prescription(prescription_id: str, user: User = Depends(require_patient), db: Session = Depends(get_db)):
    try:
        parsed = uuid.UUID(prescription_id)
    except ValueError:
        raise HTTPException(404, "No such prescription.")
    rx = prescriptions.own_visible(db, user.id, parsed)
    if rx is None:
        raise HTTPException(404, "No such prescription.")
    return prescriptions.sheet_out(db, rx)
