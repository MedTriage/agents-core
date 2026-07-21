from cerebras.cloud.sdk import Cerebras
from app.config import CEREBRAS_API_KEY, CEREBRAS_MODEL_NAME
from langsmith import traceable

client = Cerebras(api_key=CEREBRAS_API_KEY)


@traceable(name="generate_content")
def generate_content(prompt: str):
    response = client.chat.completions.create(
        model=CEREBRAS_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
