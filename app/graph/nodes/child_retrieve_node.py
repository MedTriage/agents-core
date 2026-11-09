"""Child mode: the guideline passages for the parent's question. No model call.

The previous user message rides along with the current one, so a follow-up such as "and if
he's vomiting?" still retrieves passages about the child it is about.
"""
from app.services import child_store


def child_retrieve_node(state: dict) -> dict:
    history = state.get("chat_history") or []
    previous = next((m["content"] for m in reversed(history) if m.get("role") == "user"), None)
    query = f"{previous}\n{state['user_input']}" if previous else state["user_input"]
    try:
        return {"child_passages": child_store.search(query)}
    except Exception as e:
        # No documents, or a broken index: answer without passages rather than fail the turn.
        print(e)
        return {"child_passages": [], "child_error": str(e)}
