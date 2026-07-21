# Shared data structures for agent state management
from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Per-turn input
    conversation_id: str
    user_input: str
    chat_history: Optional[list]  # [{"role": "user"|"assistant", "content": "..."}]

    # Title
    title: Optional[str]

    # Scribe: persistent per-conversation medical state, loaded/updated each turn
    scribe_output: Optional[dict]

    # Intent (binary: clinical_query | chitchat)
    intent_type: Optional[str]
    intent_confidence: Optional[float]

    # Retrieval branches — each produces the same output shape so the
    # orchestrator can treat them uniformly.
    retrieved_context: Optional[str]  # RAG's Pinecone context, for grounding checks
    rag_output: Optional[dict]
    kgrag_output: Optional[dict]
    mcp_output: Optional[dict]

    # Dempster-Shafer fusion of the branches: belief intervals, conflict K, ignorance
    # m(Θ), and the control action they imply. Computed independently of the
    # orchestrator so the two fusions can be compared; advisory, not yet routing.
    dst_output: Optional[dict]

    # Orchestrator: synthesis, conflict resolution, safety audit
    orchestrator_output: Optional[dict]
    orchestrator_decision: Optional[str]  # approve | revise | escalate
    orchestrator_response: Optional[str]
    is_emergency: Optional[bool]
    is_clarification: Optional[bool]

    # Guardian analysis
    guardian_output: Optional[dict]
    triage_level: Optional[str]

    # Per-branch re-retrieval loop. The orchestrator is the sole writer of both:
    # parallel branches must not write shared keys (LangGraph rejects concurrent
    # updates to the same channel within a superstep).
    branch_retry_counts: Optional[dict]      # {"rag": 0, "kgrag": 0, "mcp": 0}
    branch_refinement_hints: Optional[dict]  # {"rag": "...", "kgrag": None, "mcp": None}
