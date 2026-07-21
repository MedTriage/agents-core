"""Dempster-Shafer fusion of the retrieval branches — the symbolic counterpart to the
orchestrator's synthesis.

Runs at the fan-in, after every dispatched branch has completed and before the
orchestrator. Pure computation: no LLM call, no network, so it costs microseconds.

It deliberately does NOT feed the orchestrator. The orchestrator fuses the same evidence
with a language model, and the point of running both is to measure how far the two
diverge — whether the model actually combines what the branches found or just adopts the
most confident one. Showing the model this node's verdict would let it copy the answer
and destroy the measurement. The two stay independent so the comparison means something.

Advisory only at this stage: it writes `dst_output` and changes no routing.
"""
from app.services import dst
from app.graph.nodes.orchestrator_node import BRANCHES, BRANCH_STATE_KEY


def dst_node(state):
    branch_outputs = {}
    for branch in BRANCHES:
        output = state.get(BRANCH_STATE_KEY[branch])
        # A branch that never ran is absent, which is different from one that ran and
        # found nothing — only the latter is evidence about the world.
        if isinstance(output, dict):
            branch_outputs[branch] = output

    if not branch_outputs:
        return {"dst_output": None}

    try:
        fused = dst.fuse(branch_outputs)

        # Which branch to re-run, derived from the fusion rather than guessed by a
        # model. Only meaningful when ignorance is what is holding the turn back —
        # under conflict, more retrieval entrenches rather than resolves.
        fused["retry_target"] = (
            dst.most_ignorant_branch(branch_outputs)
            if fused["action"] == "retry"
            else None
        )

        return {"dst_output": fused}

    except Exception as e:
        # Fusion is advisory here, so a failure must degrade the turn rather than sink
        # it — the orchestrator carries the case either way.
        return {"dst_output": {"error": f"DST fusion failed: {str(e)}"}}
