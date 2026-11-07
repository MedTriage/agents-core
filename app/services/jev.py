"""Client for Jev, TypeSafe's decision model: it picks one of a fixed set of answers and
says how likely each one is, rather than writing text.

Reached through OpenRouter's decisions endpoint with the same key as every node's LLM call.
Errors are raised, never swallowed, so the caller can fall back to something it controls.
"""
import httpx

from app.config import JEV_MODEL, JEV_TIMEOUT_SECONDS, JEV_URL, OPENROUTER_API_KEY


def choose(state: dict, instructions: str, criteria: dict[str, str]) -> dict[str, float]:
    """Ask one choice question about `state`; return each option's probability."""
    response = httpx.post(
        JEV_URL,
        json={"model": JEV_MODEL, "state": state,
              "questions": {"complaint": {"type": "choice", "instructions": instructions,
                                          "criteria": criteria}}},
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        timeout=JEV_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["answers"]["complaint"]["probabilities"]
