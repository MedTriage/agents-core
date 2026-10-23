"""Who is calling, and may they? Every rule is enforced here, on the server. The Next.js
layer forwards the session token as a Bearer header and decides nothing."""
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.models import User, UserSession
from app.db.session import get_db
from app.services import accounts


def current_auth(request: Request, db: Session = Depends(get_db)) -> tuple[User, UserSession]:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    found = accounts.authenticate(db, token) if scheme.lower() == "bearer" and token else None
    if found is None:
        raise HTTPException(401, "Not signed in.")
    return found


def current_user(auth: tuple[User, UserSession] = Depends(current_auth)) -> User:
    return auth[0]


def require_patient(user: User = Depends(current_user)) -> User:
    if user.role != "patient":
        raise HTTPException(403, "This is a patient feature.")
    return user


def require_verified_patient(user: User = Depends(require_patient)) -> User:
    if user.email_verified_at is None:
        raise HTTPException(403, "Verify your email address first.")
    return user


def require_doctor(user: User = Depends(current_user)) -> User:
    if user.role != "doctor":
        raise HTTPException(403, "This is a doctor feature.")
    return user


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"
