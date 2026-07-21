"""Per-turn traces, one JSON line each, for offline analysis.

The orchestrator fuses the retrieval branches with a language model and `dst_node`
fuses the same evidence arithmetically, independently. Every turn therefore produces a
paired observation — what the model concluded versus what the evidence supports — and
this is what records it. Without persistence those pairs exist only in whatever terminal
the request came back in.

Because the orchestrator never sees the fusion, this keeps producing valid paired data
in normal operation rather than only during a dedicated evaluation run.

Tracing is strictly observational: every failure here is swallowed. A turn must never
fail because a log line could not be written.
"""
import json
import os
import time

from app.config import TRACE_ENABLED, TRACE_PATH
from app.graph.nodes.orchestrator_node import BRANCHES, BRANCH_STATE_KEY


def _branch_summary(output) -> dict:
    """What a branch concluded and what it actually retrieved, side by side."""
    if not isinstance(output, dict):
        return {"status": "absent"}
    if "error" in output:
        return {"status": "error", "error": output["error"]}

    return {
        "status": "ok",
        "probable_diagnosis": output.get("probable_diagnosis"),
        "differentials": output.get("differentials") or [],
        # The branch's own confidence — recorded so it can be compared against the
        # retrieval-derived mass, never used to compute one.
        "self_reported_confidence": output.get("confidence"),
        "retrieval_signal": output.get("retrieval_signal"),
    }


def record(state: dict, guardian_delta: dict) -> None:
    if not TRACE_ENABLED:
        return

    try:
        orchestrator_output = state.get("orchestrator_output") or {}
        dst_output = state.get("dst_output") or {}
        guardian_output = guardian_delta.get("guardian_output") or {}

        line = {
            "timestamp": time.time(),
            "conversation_id": state.get("conversation_id"),
            "user_input": state.get("user_input"),
            "intent_type": state.get("intent_type"),
            "branches": {
                b: _branch_summary(state.get(BRANCH_STATE_KEY[b])) for b in BRANCHES
            },
            "dst": {
                "frame": dst_output.get("frame"),
                "conflict": dst_output.get("conflict"),
                "ignorance": dst_output.get("ignorance"),
                "top_hypothesis": dst_output.get("top_hypothesis"),
                "hypotheses": dst_output.get("hypotheses"),
                "unruled_out": dst_output.get("unruled_out"),
                "action": dst_output.get("action"),
                "retry_target": dst_output.get("retry_target"),
            },
            "orchestrator": {
                "probable_diagnosis": orchestrator_output.get("probable_diagnosis"),
                "decision": orchestrator_output.get("decision"),
                "safety_risk": orchestrator_output.get("safety_risk"),
                "confidence_adjusted": orchestrator_output.get("confidence_adjusted"),
                "evidence_grounded": orchestrator_output.get("evidence_grounded"),
                "is_emergency": orchestrator_output.get("is_emergency"),
                "response": state.get("orchestrator_response"),
            },
            "guardian": {
                "triage_level": guardian_delta.get("triage_level"),
                "reasoning": guardian_output.get("reasoning"),
            },
            "dst_audit": guardian_output.get("dst_audit"),
        }

        os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")

    except Exception:
        # Observation must not have side effects on the thing being observed.
        pass
