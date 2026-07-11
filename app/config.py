# Configurations for handling environment variables and other settings for the application
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

MODEL_NAME = "gemini-2.5-flash"
OPENAI_MODEL_NAME = "gpt-4o-mini"
CEREBRAS_MODEL_NAME = "gpt-oss-120b"
# Embeddings stay on OpenAI — Cerebras serves generation only, not embeddings,
# so OPENAI_API_KEY is still required for the RAG pipeline.
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

# Re-retrieval loop (orchestrator → retrieval branch retry)
# Applied per branch: the orchestrator may send each of rag/kgrag/mcp back
# this many times independently.
MAX_BRANCH_RETRIES = 1

# Scribe per-conversation state files
CONVERSATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "conversations"
)

# Medical MCP server (JamesANZ/medical-mcp, vendored — source only, node_modules and
# build/ are gitignored). Must be built before the MCP branch works:
#   cd vendor/medical-mcp && npm install && npm run build
MCP_SERVER_DIR = os.getenv(
    "MCP_SERVER_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "vendor", "medical-mcp"
    ),
)
# Whole-session budget: process spawn + handshake + one tool call. Live sources are
# slow (FDA ~3s, PubMed ~11s, Google Scholar scrapes with puppeteer and is slower),
# so this caps how long the MCP branch can stall the graph.
MCP_TIMEOUT = 45

# SNOMED CT knowledge graph (KGRAG branch).
# Built by `python -m app.rag.snomed_ingest <path-to-RF2-Snapshot>`.
SNOMED_DB_PATH = os.getenv(
    "SNOMED_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "snomed.db"),
)
# How many concepts the entity-mapping step may resolve from one query, and how far
# up/down the IS_A hierarchy to walk from each. Depth is bounded deliberately —
# SNOMED's upper hierarchy generalises fast ("disorder" → "clinical finding"), so
# deep walks add noise, not evidence.
SNOMED_MAX_CONCEPTS = 6
SNOMED_MAX_DEPTH = 2
# Hard cap on relatives returned per traversal. A general concept ("pain") has thousands
# of descendants in the real release; without this the subgraph would swamp the prompt.
SNOMED_MAX_RELATIVES = 25
# How many differentials get enriched with their own defining attributes (causative
# agent, finding site). These are what let the agent say *how* to rule a differential out.
SNOMED_MAX_ENRICHED_SUBTYPES = 8

# Embedding batch size
EMBEDDING_BATCH_SIZE = 100

# Pinecone upsert batch size
PINECONE_UPSERT_BATCH_SIZE = 100

# Max tokens for embedding model (text-embedding-3-small limit)
EMBEDDING_MAX_TOKENS = 8191