"""Health readings the patient enters on their dashboard. The limits only catch typos: they say
nothing about what is healthy, and no reading is ever judged normal or high here."""
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import HealthReading


@dataclass(frozen=True)
class Kind:
    label: str
    unit: str
    low: float
    high: float
    decimals: int
    # Blood pressure only: the second number's limits.
    low2: float | None = None
    high2: float | None = None


# Order is the order the agents read them in.
KINDS = {
    "weight": Kind("weight", "kg", 1, 500, 1),
    "height": Kind("height", "cm", 30, 272, 1),
    "blood_pressure": Kind("blood pressure", "mmHg", 40, 300, 0, low2=20, high2=200),
    "pulse": Kind("pulse", "bpm", 20, 300, 0),
}


def check(kind: str, value: float, value2: float | None) -> tuple[float, float | None]:
    """The values to store, rounded to the kind's precision. Raises ValueError with a sentence
    for the patient."""
    spec = KINDS[kind]
    if not spec.low <= value <= spec.high:
        raise ValueError(f"{spec.label.capitalize()} should be between {spec.low} and {spec.high} {spec.unit}")
    if spec.low2 is None:
        if value2 is not None:
            raise ValueError(f"{spec.label.capitalize()} takes one number")
        return round(value, spec.decimals), None
    if value2 is None:
        raise ValueError("Blood pressure needs both numbers")
    if not spec.low2 <= value2 <= spec.high2:
        raise ValueError(f"The lower number should be between {spec.low2} and {spec.high2} {spec.unit}")
    if value2 >= value:
        raise ValueError("The first number should be higher than the second")
    return round(value, spec.decimals), round(value2, spec.decimals)


def on_day(day: date) -> str:
    return f"{day.day} {day:%b %Y}"


def _number(value) -> str:
    """72.0 reads as 72, 72.5 as 72.5."""
    return f"{float(value):g}"


def reading_out(reading: HealthReading) -> dict:
    return {
        "id": reading.id,
        "kind": reading.kind,
        "value": float(reading.value),
        "value2": float(reading.value2) if reading.value2 is not None else None,
        "unit": KINDS[reading.kind].unit,
        "measured_on": reading.measured_on.isoformat(),
        "created_at": reading.created_at.isoformat(),
    }


def of_patient(db: Session, user_id) -> list[HealthReading]:
    """Newest measured first; among one day's readings, the last entered first."""
    return list(db.scalars(select(HealthReading).where(HealthReading.user_id == user_id)
                           .order_by(HealthReading.measured_on.desc(), HealthReading.id.desc())))


def background_lines(db: Session, user_id) -> list[str]:
    """The latest of each kind, for the patient background. Empty when there are none, so a
    patient with no readings gets exactly the background they had before."""
    latest = {}
    for reading in of_patient(db, user_id):
        latest.setdefault(reading.kind, reading)
    lines = []
    for kind, spec in KINDS.items():
        reading = latest.get(kind)
        if reading is None:
            continue
        shown = _number(reading.value)
        if reading.value2 is not None:
            shown += f"/{_number(reading.value2)}"
        lines.append(f"- Latest {spec.label}: {shown} {spec.unit} (recorded {on_day(reading.measured_on)})")
    return ["Recorded by the patient on their dashboard:", *lines] if lines else []
