"""Test harness: a real Postgres test database, the real FastAPI app, a fake triage graph.

The graph is the only thing faked. It is the slow, paid, external part (nine LLM calls a
turn), and everything these tests protect — who may read what, what gets persisted, what
an email may contain — sits around it, not inside it.
"""
import os
import re
from urllib.parse import parse_qs, urlparse

from dotenv import dotenv_values

# Point the app at the test database BEFORE anything imports app.config.
_env = dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))
_base_url = os.environ.get("DATABASE_URL") or _env.get("DATABASE_URL")
if not _base_url:
    raise RuntimeError("DATABASE_URL is not set; see README for starting the database")
os.environ["DATABASE_URL"] = re.sub(r"/[^/]+$", "/medtriage_test", _base_url)
os.environ["EMAIL_WORKER_ENABLED"] = "0"
# Safety net: a test that accidentally reaches the real graph must fail, not spend money.
# load_dotenv() never overrides a variable that is already set.
os.environ["OPENROUTER_API_KEY"] = "test-key-no-network"

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    """Migrate a clean schema once per run, through the real migrations."""
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    command.upgrade(cfg, "head")
    yield


@pytest.fixture(autouse=True)
def _clean_tables():
    with engine.begin() as conn:
        tables = conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' "
            "AND tablename <> 'alembic_version'"
        )).scalars().all()
        if tables:
            conn.execute(text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"))
    yield


class FakeGraph:
    """Stands in for the compiled LangGraph. Records every input state it is given and
    returns a canned final state shaped like the real one."""

    def __init__(self):
        self.inputs = []
        self.level = "level_1"
        self.response = "Rest and drink fluids."
        self.scribe_record = None
        self.title = "Sore Throat"

    def _result(self, state):
        record = self.scribe_record or {
            "past_conditions": [], "symptoms": [], "medications": [], "allergies": [],
            "demographics": {}, "notes": [], "turn": int((state.get("scribe_output") or {}).get("turn") or 0) + 1,
        }
        return {
            **state,
            "title": self.title,
            "intent_type": "clinical_query",
            "intent_confidence": 0.9,
            "scribe_output": record,
            "parametric_output": {"probable_diagnosis": "Common cold", "differentials": [],
                                  "recommended_actions": [], "citations": [], "confidence": 0.8},
            "kgrag_output": None,
            "mcp_output": None,
            "dst_output": None,
            "orchestrator_output": {"response": self.response, "decision": "approve",
                                    "safety_risk": "low", "confidence_adjusted": 0.8,
                                    "probable_diagnosis": "Common cold", "differentials": []},
            "orchestrator_decision": "approve",
            "orchestrator_response": self.response,
            "is_emergency": self.level == "level_3",
            "is_clarification": False,
            "guardian_output": {"triage_level": self.level, "reasoning": "test",
                                "requires_doctor": self.level != "level_1",
                                "ai_lock": self.level == "level_3"},
            "triage_level": self.level,
        }

    def invoke(self, state):
        self.inputs.append(state)
        return self._result(state)

    def stream(self, state, stream_mode="updates"):
        self.inputs.append(state)
        result = self._result(state)
        yield {"scribe_node": {"scribe_output": result["scribe_output"]}}
        yield {"orchestrator_node": {k: result[k] for k in (
            "orchestrator_output", "orchestrator_decision", "orchestrator_response",
            "is_emergency", "is_clarification")}}
        yield {"guardian_node": {"guardian_output": result["guardian_output"],
                                 "triage_level": result["triage_level"]}}


@pytest.fixture
def graph():
    return FakeGraph()


@pytest.fixture
def client(graph):
    app.state.graph = graph
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


# ── helpers ──────────────────────────────────────────────────────────────────────────

def outbox(template=None, to=None):
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT recipient, template, payload, status, attempts FROM email_outbox ORDER BY id"
        )).mappings().all()
    return [dict(r) for r in rows
            if (template is None or r["template"] == template) and (to is None or r["recipient"] == to)]


def link_token(email_row) -> str:
    """The single-use token carried in an emailed link."""
    return parse_qs(urlparse(email_row["payload"]["link"]).query)["token"][0]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


PASSWORD = "correct horse battery"


def signup(client, email="pat@example.com", password=PASSWORD, name="Pat"):
    return client.post("/auth/signup", json={"email": email, "password": password,
                                             "name": name, "consent": True})


def patient(client, email="pat@example.com", password=PASSWORD, name="Pat") -> str:
    """A verified, logged-in patient. Returns the session token."""
    assert signup(client, email, password, name).status_code == 201
    token = link_token(outbox("verify_email", to=email)[-1])
    assert client.post("/auth/verify", json={"token": token}).status_code == 200
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def doctor(client, email="doc@example.com", password=PASSWORD, name="Dr Rao") -> str:
    """An invited doctor who has accepted. Returns the session token."""
    from app.admin import invite_doctor
    with SessionLocal() as s:
        invite_doctor(s, email, name)
    token = link_token(outbox("doctor_invite", to=email)[-1])
    assert client.post("/auth/invite/accept", json={"token": token, "password": password}).status_code == 200
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]
