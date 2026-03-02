# Graph builder
from langgraph.graph import StateGraph
from app.graph.state import AgentState

from app.graph.nodes.intent_node import intent_node
from app.graph.nodes.rag_node import rag_node
from app.graph.nodes.image_node import image_node
from app.graph.nodes.companion_node import companion_node
from app.graph.nodes.critic_node import critic_node


def route_after_intent(state: AgentState):
    intent = state.get("intent_type")

    if intent == "emergency":
        return "rag_node"
    elif intent == "clinical_query":
        return "rag_node"
    elif intent == "image_input":
        return "image_node"
    elif intent == "chitchat":
        return "companion_node"
    else:
        return "rag_node"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("intent_router", intent_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("image_node", image_node)
    builder.add_node("companion_node", companion_node)
    builder.add_node("critic_node", critic_node)

    builder.set_entry_point("intent_router")

    builder.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "rag_node": "rag_node",
            "image_node": "image_node",
            "companion_node": "companion_node",
        },
    )

    builder.add_edge("rag_node", "critic_node")

    builder.set_finish_point("critic_node")
    builder.set_finish_point("image_node")
    builder.set_finish_point("companion_node")

    return builder.compile()
