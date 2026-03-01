# Configurations for handling environment variables and other settings for the application
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

MODEL_NAME = "gemini-2.5-flash" 
OPENAI_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# RAG settings
RETRIEVAL_TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.35

# Chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retry settings
MAX_RETRIES = 2
REQUEST_TIMEOUT = 15

# Embedding batch size
EMBEDDING_BATCH_SIZE = 100

# Pinecone upsert batch size
PINECONE_UPSERT_BATCH_SIZE = 100

# Max tokens for embedding model (text-embedding-3-small limit)
EMBEDDING_MAX_TOKENS = 8191