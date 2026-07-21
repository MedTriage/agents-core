# Main application entry point
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.graph.builder import build_graph
from app.services.explain import build_explanation

app = FastAPI()

graph = build_graph()

# Retrieval branches, in the order their anonymous slots appear to the client. The node
# names stay server-side; what crosses the wire is an index and a status.
_BRANCH_NODES = ("rag_node", "kgrag_node", "mcp_node")

class InputRequest(BaseModel):
    conversation_id: str
    text: str
    chat_history: list[dict] = []

# Home route
@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

# Input route
@app.post("/process")
def process_input(request: InputRequest):
    result = graph.invoke({
        "conversation_id": request.conversation_id,
        "user_input": request.text,
        "chat_history": request.chat_history
    })
    return {**result, "explanation": build_explanation(result)}


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


@app.post("/process/stream")
def process_stream(request: InputRequest):
    """Server-sent events version of /process.

    The pipeline takes tens of seconds — the slow part being live external sources — and
    a blocking request spends all of it silent. Streaming does not make it faster; it
    makes the time legible, and it keeps bytes moving so a proxy or serverless platform
    does not kill a request that is still working.

    The terminal `done` event carries exactly the /process payload, so the client has
    one shape to handle rather than two.
    """
    def events():
        accumulated: dict = {}

        try:
            for chunk in graph.stream(
                {
                    "conversation_id": request.conversation_id,
                    "user_input": request.text,
                    "chat_history": request.chat_history,
                },
                stream_mode="updates",
            ):
                for node, update in (chunk or {}).items():
                    if isinstance(update, dict):
                        accumulated.update(update)

                    event = _progress_event(node, accumulated)
                    if event:
                        yield f"data: {json.dumps(event)}\n\n"

            yield "data: " + json.dumps({
                "type": "done",
                **accumulated,
                "explanation": build_explanation(accumulated),
            }) + "\n\n"

        except Exception as e:
            # The stream has already begun, so an HTTP error status is no longer
            # available. Fail as a typed event the client can render.
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Proxies that buffer will defeat the entire point of streaming.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
