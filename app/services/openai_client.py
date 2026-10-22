"""Chat client for every node's LLM call.

Points at OpenRouter, which speaks the OpenAI wire protocol — hence the OpenAI SDK and the
module name.
"""
import threading

from openai import BadRequestError, OpenAI
from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL_NAME,
    OPENROUTER_PROVIDER,
    OPENROUTER_SEED,
    OPENROUTER_TEMPERATURE,
)
from langsmith import traceable

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# Decoding parameters the deployed model turned out not to accept. Reasoning models in the
# GPT-5 family reject an explicit temperature, and which parameters a given model takes is
# not knowable without asking it, so the first rejection retires the offending parameter
# for the rest of the process. Losing determinism is better than failing every call.
_unsupported = set()
_unsupported_lock = threading.Lock()


def _decoding_params() -> dict:
    with _unsupported_lock:
        retired = set(_unsupported)
    params = {"temperature": OPENROUTER_TEMPERATURE, "seed": OPENROUTER_SEED}
    return {k: v for k, v in params.items() if v is not None and k not in retired}


def _retire(param: str) -> None:
    with _unsupported_lock:
        _unsupported.add(param)


@traceable(name="generate_content")
def generate_content(prompt: str):
    while True:
        params = _decoding_params()
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                # Routing is an OpenRouter extension, so it rides outside the documented
                # request body. Kept out of `params` deliberately: the retire loop below
                # exists for decoding parameters an upstream model rejects, and must never
                # quietly drop the provider pin on an unrelated error.
                extra_body={"provider": OPENROUTER_PROVIDER},
                **params,
            )
            break
        except BadRequestError as e:
            # Only a complaint about a decoding parameter is recoverable here. Anything
            # else is a real bad request and must surface.
            offending = next((p for p in params if p in str(e)), None)
            if offending is None:
                raise
            _retire(offending)

    return response.choices[0].message.content
