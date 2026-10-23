"""Admin tasks, command line only — there is deliberately no admin web UI.

    python -m app.admin invite-doctor doctor@example.com --name "Dr Rao"
"""
import argparse

from app.db.session import SessionLocal
from app.services.accounts import EmailTaken, invite_doctor

__all__ = ["invite_doctor"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.admin")
    sub = parser.add_subparsers(dest="command", required=True)
    invite = sub.add_parser("invite-doctor", help="create a doctor account and email an invitation")
    invite.add_argument("email")
    invite.add_argument("--name", required=True)
    args = parser.parse_args()

    with SessionLocal() as db:
        try:
            user = invite_doctor(db, args.email, args.name)
        except EmailTaken:
            parser.exit(1, f"{args.email} already has a patient account.\n")
    print(f"Invitation queued for {user.email}. It is sent by the API's email worker.")


if __name__ == "__main__":
    main()
