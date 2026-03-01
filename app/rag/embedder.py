import tiktoken
from openai import OpenAI
from app.config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_MAX_TOKENS

client = OpenAI(api_key=OPENAI_API_KEY)


_tokenizer = tiktoken.get_encoding("cl100k_base")

def _truncate_to_token_limit(text: str, max_tokens: int = EMBEDDING_MAX_TOKENS) -> str:
    """Truncate text to fit within the embedding model's token limit."""
    tokens = _tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return _tokenizer.decode(tokens)
    return text


def embed_text(text: str) -> list[float]:
    """Embed a single text string. Truncates if it exceeds token limit."""
    text = _truncate_to_token_limit(text)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts in a single API call.
    Truncates each text to fit within the token limit.
    Returns embeddings in the same order as the input texts.
    """
    truncated = [_truncate_to_token_limit(t) for t in texts]
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=truncated
    )
    # Sort by index to ensure order matches input
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]