# Graph builder
from langgraph.graph import StateGraph
from app.graph.state import AgentState

from app.graph.nodes.scribe_node import scribe_node
from app.graph.nodes.intent_node import intent_node
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.kgrag_node import kgrag_node
from app.graph.nodes.mcp_node import mcp_node
from app.graph.nodes.dst_node import dst_node
from app.graph.nodes.orchestrator_node import orchestrator_node
from app.graph.nodes.guardian_node import guardian_node

# Retrieval branches that fan out from the intent router and fan back in to the
# orchestrator. Keys match the branch names the orchestrator uses in its hints.
BRANCH_NODES = {
    "rag": "rag_node",
    "kgrag": "kgrag_node",
    "mcp": "mcp_node",
}


def route_after_intent(state: AgentState):
    """Chitchat goes straight to the orchestrator; anything clinical fans out to all
    three retrieval branches in parallel."""
    if state.get("intent_type") == "chitchat":
        return ["orchestrator_node"]

    return list(BRANCH_NODES.values())


def route_after_orchestrator(state: AgentState):
    """Re-run only the branches the orchestrator flagged.

    The orchestrator has already applied the per-branch retry budget and excluded
    branches that produced no evidence, so a hint here is always actionable. No hints
    means the case is done being retried and moves on to triage.
    """
    hints = state.get("branch_refinement_hints") or {}
    retry_nodes = [BRANCH_NODES[b] for b, hint in hints.items() if hint and b in BRANCH_NODES]

    return retry_nodes or ["guardian_node"]


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("scribe_node", scribe_node)
    builder.add_node("intent_router", intent_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("kgrag_node", kgrag_node)
    builder.add_node("mcp_node", mcp_node)
    builder.add_node("dst_node", dst_node)
    builder.add_node("orchestrator_node", orchestrator_node)
    builder.add_node("guardian_node", guardian_node)

    builder.set_entry_point("scribe_node")
    builder.add_edge("scribe_node", "intent_router")

    builder.add_conditional_edges(
        "intent_router",
        route_after_intent,
        [*BRANCH_NODES.values(), "orchestrator_node"],
    )

    # Fan-in: dst_node runs once, after every dispatched branch has completed, and the
    # orchestrator follows it. Sequencing them costs nothing — the fusion is arithmetic,
    # no LLM call — and it keeps dst_node a single writer of `dst_output` rather than a
    # fourth participant in the branches' superstep.
    #
    # The chitchat path skips both branches and dst_node, going straight to the
    # orchestrator: there is no retrieved evidence to fuse.
    for node in BRANCH_NODES.values():
        builder.add_edge(node, "dst_node")

    builder.add_edge("dst_node", "orchestrator_node")

    builder.add_conditional_edges(
        "orchestrator_node",
        route_after_orchestrator,
        [*BRANCH_NODES.values(), "guardian_node"],
    )

    builder.set_finish_point("guardian_node")

    return builder.compile()
