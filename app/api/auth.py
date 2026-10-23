from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.orm import Session

from app.api.deps import client_ip, current_auth
from app.config import PASSWORD_MIN_LENGTH
from app.db.models import User
from app.db.session import get_db
from app.services import accounts

router = APIRouter(prefix="/auth", tags=["auth"])

Password = Field(min_length=PASSWORD_MIN_LENGTH, max_length=256)

# Identical for every outcome, so these endpoints cannot be used to find out who has an
# account.
ACCEPTED = {"status": "ok"}


def user_out(user: User) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "verified": user.email_verified_at is not None,
    }


class SignupIn(BaseModel):
    email: EmailStr
    password: str = Password
    name: str = Field(min_length=1, max_length=200)
    consent: bool

    @field_validator("consent")
    @classmethod
    def _must_consent(cls, value: bool) -> bool:
        if not value:
            raise ValueError("consent is required to create an account")
        return value


class TokenIn(BaseModel):
    token: str = Field(min_length=1, max_length=200)


class TokenPasswordIn(TokenIn):
    password: str = Password


class EmailIn(BaseModel):
    email: EmailStr


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=256)


@router.post("/signup", status_code=201)
def signup(body: SignupIn, db: Session = Depends(get_db)):
    try:
        user = accounts.signup(db, body.email, body.password, body.name)
    except accounts.EmailTaken:
        raise HTTPException(409, "An account with this email already exists.")
    return {"id": str(user.id)}


@router.post("/verify")
def verify(body: TokenIn, db: Session = Depends(get_db)):
    if not accounts.verify_email(db, body.token):
        raise HTTPException(400, "This link is invalid or has expired.")
    return {"status": "verified"}


@router.post("/verify/resend", status_code=202)
def resend_verification(body: EmailIn, db: Session = Depends(get_db)):
    accounts.resend_verification(db, body.email)
    return ACCEPTED


@router.post("/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    try:
        token, user = accounts.login(db, body.email, body.password, client_ip(request))
    except accounts.RateLimited:
        raise HTTPException(429, "Too many failed attempts. Try again in 15 minutes.")
    except accounts.InvalidCredentials:
        raise HTTPException(401, "Incorrect email or password.")
    return {"token": token, "user": user_out(user)}


@router.post("/logout", status_code=204)
def logout(auth=Depends(current_auth), db: Session = Depends(get_db)):
    accounts.logout(db, auth[1])
    return Response(status_code=204)


@router.post("/password/forgot", status_code=202)
def forgot_password(body: EmailIn, db: Session = Depends(get_db)):
    accounts.request_password_reset(db, body.email)
    return ACCEPTED


@router.post("/password/reset")
def reset_password(body: TokenPasswordIn, db: Session = Depends(get_db)):
    if not accounts.reset_password(db, body.token, body.password):
        raise HTTPException(400, "This link is invalid or has expired.")
    return {"status": "password_changed"}


@router.post("/invite/accept")
def accept_invite(body: TokenPasswordIn, db: Session = Depends(get_db)):
    if not accounts.accept_invite(db, body.token, body.password):
        raise HTTPException(400, "This invitation is invalid or has expired.")
    return {"status": "accepted"}
