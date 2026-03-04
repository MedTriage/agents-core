from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL_NAME
from langsmith import traceable

client = OpenAI(api_key=OPENAI_API_KEY)


@traceable(name="generate_content")
def generate_content(prompt: str):
    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
