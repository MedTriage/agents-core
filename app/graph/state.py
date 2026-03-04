# Shared data structures for agent state management
from typing import TypedDict, Optional

class AgentState(TypedDict):
    user_input: str
    chat_history: Optional[list]  # [{"role": "user"|"assistant", "content": "..."}]

    # Title
    title: Optional[str]

    # Intent
    intent_type: Optional[str]
    intent_confidence: Optional[float]

    # Retrieved context (passed from RAG to Critic for verification)
    retrieved_context: Optional[str]

    # Downstream outputs
    rag_output: Optional[dict]
    image_output: Optional[str]
    companion_output: Optional[str]

    # Critic feedback
    critic_output: Optional[dict]
    critic_decision: Optional[str]
    critic_response: Optional[str]

    # Guardian analysis
    guardian_output: Optional[dict]
    triage_level: Optional[str]

    # Re-retrieval loop (critic → RAG)
    rag_retry_count: Optional[int]
    critic_refinement_hint: Optional[str]