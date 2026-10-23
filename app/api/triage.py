"""Triage turns. The server owns the conversation: it checks the caller may post, supplies
the history and the patient background, and saves the outcome. The client sends only the
text and, after the first turn, the conversation id the server gave it."""
import json
import logging
import queue
import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import require_verified_patient
from app.db.models import User
from app.db.session import get_db
from app.services import turns
from app.services.explain import build_explanation

log = logging.getLogger(__name__)

router = APIRouter(tags=["triage"])

# Retrieval branches, in the order their anonymous slots appear to the client. The node
# names stay server-side; what crosses the wire is an index and a status.
# Order must match explain._BRANCHES — the index is what crosses the wire, so a mismatch
# silently attributes one source's status to another's slot.
_BRANCH_NODES = ("parametric_node", "kgrag_node", "mcp_node")

# Shown to the patient when a turn fails. The real cause goes to the server log only.
_FAILED = "The assessment could not be completed. Please try again."


class TurnIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None


def _start(db: Session, user: User, body: TurnIn) -> turns.Turn:
    try:
        return turns.prepare_turn(db, user, body.conversation_id, body.text)
    except turns.TurnRefused as refused:
        raise HTTPException(refused.status, refused.detail)


@router.post("/process")
def process_input(body: TurnIn, request: Request, user: User = Depends(require_verified_patient),
                  db: Session = Depends(get_db)):
    turn = _start(db, user, body)
    conversation_id = turn.state["conversation_id"]
    try:
        result = request.app.state.graph.invoke(turn.state)
        return turns.finish_turn(turn, result)
    except Exception:
        log.exception("triage turn failed (conversation %s)", conversation_id)
        turns.release(conversation_id)
        raise HTTPException(502, _FAILED)


def _progress_event(node: str, accumulated: dict) -> dict | None:
    """Map a completed node onto a patient-safe progress event, or nothing.

    Only three things are worth telling the patient about while they wait, and none of
    them names a component: that their message was read, that a source finished, and
    that the sources were weighed against each other.
    """
    if node == "scribe_node":
        return {"type": "stage", "stage": "reading"}

    if node in _BRANCH_NODES:
        index = _BRANCH_NODES.index(node)
        explanation = build_explanation(accumulated)
        return {
            "type": "source",
            "index": index,
            "status": explanation["sources"][index]["status"],
            # Carried the moment the ontology walk lands, well before the fusion. Every
            # candidate is still undecided here, so the interface can show the real
            # shortlist being weighed instead of a spinner.
            "differential": explanation["differential"],
        }

    if node == "dst_node":
        explanation = build_explanation(accumulated)
        return {
            "type": "fusion",
            "agreement": explanation["agreement"],
            "certainty": explanation["certainty"],
            "differential": explanation["differential"],
        }

    return None


@router.post("/process/stream")
def process_stream(body: TurnIn, request: Request, user: User = Depends(require_verified_patient),
                   db: Session = Depends(get_db)):
    """Server-sent events version of /process.

    The pipeline takes tens of seconds — the slow part being live external sources — and
    a blocking request spends all of it silent. Streaming does not make it faster; it
    makes the time legible, and it keeps bytes moving so a proxy does not kill a request
    that is still working.

    The turn runs in its own thread and is saved there, so a patient who closes the tab
    mid-turn still finds the answer on their next visit. Refusals (locked, busy, over the
    limit) happen before the stream opens, so they arrive as real HTTP statuses.

    The terminal `done` event carries exactly the /process payload, so the client has
    one shape to handle rather than two.
    """
    turn = _start(db, user, body)
    state = turn.state
    graph = request.app.state.graph
    events: queue.Queue = queue.Queue()

    def run():
        accumulated = dict(state)
        try:
            for chunk in graph.stream(state, stream_mode="updates"):
                for node, update in (chunk or {}).items():
                    if isinstance(update, dict):
                        accumulated.update(update)
                    event = _progress_event(node, accumulated)
                    if event:
                        events.put(event)
            events.put({"type": "done", **turns.finish_turn(turn, accumulated)})
        except Exception:
            log.exception("streamed triage turn failed (conversation %s)", state["conversation_id"])
            turns.release(state["conversation_id"])
            events.put({"type": "error", "message": _FAILED})
        finally:
            events.put(None)

    threading.Thread(target=run, name="triage-turn", daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # Proxies that buffer will defeat the entire point of streaming.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
