"""Child-health mode's graph: Q&A from IMNCI/WHO documents, with no triage.

    child_signs_node → child_retrieve_node → child_answer_node

Separate from the triage graph, so the measured prompts and nodes stay untouched. The sign
check and the retrieval don't depend on each other; they run in sequence because neither
takes long (one Jev call, one local search).
"""
from langgraph.graph import StateGraph

from app.graph.nodes.child_answer_node import child_answer_node
from app.graph.nodes.child_retrieve_node import child_retrieve_node
from app.graph.nodes.child_signs_node import child_signs_node
from app.graph.state import ChildState


def build_child_graph():
    builder = StateGraph(ChildState)
    builder.add_node("child_signs_node", child_signs_node)
    builder.add_node("child_retrieve_node", child_retrieve_node)
    builder.add_node("child_answer_node", child_answer_node)

    builder.set_entry_point("child_signs_node")
    builder.add_edge("child_signs_node", "child_retrieve_node")
    builder.add_edge("child_retrieve_node", "child_answer_node")
    builder.set_finish_point("child_answer_node")

    return builder.compile()
