import json
from app.services.gemini_client import generate_content
from app.services.retry import retry_on_exception

INTENT_PROMPT = """
You are a strict medical system intent classifier.

Your task is to classify user input into EXACTLY ONE of the following categories:

1. clinical_query  → Symptoms, medical concerns, diagnosis requests, medication questions, health advice.
2. image_input     → Mentions of medical images, scans, X-rays, MRI, CT, ultrasound, lab reports.
3. chitchat        → Greetings, casual conversation, non-medical talk.

Rules:
- Return ONLY valid JSON.
- Do NOT include explanations.
- Do NOT include extra keys.
- Do NOT change label names.
- Confidence must be a float between 0 and 1.
- If uncertain between categories, choose "clinical_query".

Output format (STRICT):

{
  "type": "clinical_query | image_input | chitchat",
  "confidence": 0.0
}

--------------------------------
Examples:

Input: I have had fever for 3 days and body pain
Output:
{
  "type": "clinical_query",
  "confidence": 0.97
}

Input: Here is my chest X-ray image for review
Output:
{
  "type": "image_input",
  "confidence": 0.98
}

Input: Good morning doctor
Output:
{
  "type": "chitchat",
  "confidence": 0.96
}

--------------------------------

Now classify the following input.

Input:
"""

@retry_on_exception
def call_model(prompt: str):
    return generate_content(prompt)

def intent_node(state):
    user_input = state["user_input"]
    prompt = INTENT_PROMPT + user_input

    try:
        raw_output = call_model(prompt)

        # JSON extraction
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model output")

        cleaned = raw_output[start:end]

        parsed = json.loads(cleaned)

        intent_type = parsed["type"]
        confidence = float(parsed["confidence"])

        if intent_type not in [
            "clinical_query",
            "image_input",
            "chitchat"
        ]:
            raise ValueError("Invalid intent type returned")

        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence out of range")

        state["intent_type"] = intent_type
        state["intent_confidence"] = confidence

    except Exception as e:
        # Fallback
        print(e)
        state["intent_type"] = "clinical_query"
        state["intent_confidence"] = 0.0
        state["intent_error"] = str(e)

    return state