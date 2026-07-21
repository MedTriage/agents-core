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
# Concepts any ONE extracted term may claim out of that total. Without a per-term cap a
# single common word ("fever") fills the whole budget with its incidental matches and
# the remaining terms — often the actual chief complaint — never link at all.
SNOMED_MAX_CONCEPTS_PER_TERM = 2
SNOMED_MAX_DEPTH = 2
# Hard cap on relatives returned per traversal. A general concept ("pain") has thousands
# of descendants in the real release; without this the subgraph would swamp the prompt.
SNOMED_MAX_RELATIVES = 25
# How many differentials get enriched with their own defining attributes (causative
# agent, finding site). These are what let the agent say *how* to rule a differential out.
SNOMED_MAX_ENRICHED_SUBTYPES = 8

# --- Dempster-Shafer evidence fusion (app/services/dst.py) ---
# EVERY constant below is a CALIBRATION PARAMETER, not a tuned truth. They are
# placeholders pending a fit on a held-out query set, and any result reported from
# them must carry a sensitivity analysis. Hand-picking these and presenting the
# output as principled would simply relocate the magic constant.
#
# RAG: similarity below RETRIEVAL_SCORE_THRESHOLD never reaches the branch, so that
# threshold is the natural zero. Saturation is where more similarity stops meaning
# more evidence.
DST_RAG_SATURATION = 0.60
# Corroboration by document count modulates the score rather than dominating it: a
# single strong chunk is still evidence. Floor is the multiplier at n=1.
DST_COVERAGE_FLOOR = 0.5
# KGRAG: FTS5 bm25 is negative, more negative is better. Magnitude at which a match
# counts as unambiguous.
DST_BM25_SATURATION = 5.0
# A concept's evidential weight falls as its subtype set grows — matching "pain"
# locates the patient in the ontology but licenses almost no belief. Scale is the
# subtype count at which strength halves.
DST_SPECIFICITY_SCALE = 10.0
# MCP: source authority by tool. An FDA label is not a Google Scholar scrape.
DST_MCP_TIER = {
    "fda": 0.9, "drug": 0.9, "rxnorm": 0.9,
    "who": 0.85, "health-statistics": 0.85,
    "pubmed": 0.8, "article": 0.8,
    "scholar": 0.5,
}
DST_MCP_TIER_DEFAULT = 0.6
# Evidence volume at which an MCP result counts as substantive rather than a
# technically-successful empty hit.
DST_MCP_VOLUME_SCALE = 500.0
# Shafer discount per source (β). Handles both source reliability and the fact that
# the branches are NOT independent — one model synthesizes all three, and the
# corpora overlap. Dempster's rule assumes independence; discounting is the standard
# concession to its absence, and this is an ablation axis, not a fixed truth.
DST_DISCOUNT = {"rag": 0.9, "kgrag": 0.85, "mcp": 0.7}
# Control thresholds on the (conflict, ignorance) pair. Conflict means the sources
# disagree — re-retrieval entrenches rather than resolves, so it escalates. Ignorance
# means nobody committed — that is what re-retrieval actually fixes.
# Diagnosis labels are canonicalised through SNOMED so that two branches naming the same
# disease differently ("COVID-19" vs "Disease caused by SARS-CoV-2") form ONE hypothesis.
# A resolution is only accepted when this fraction of the label's words appear in the
# SNOMED description that matched — otherwise a vague label would be canonicalised onto
# whatever bm25 happened to rank first, silently merging distinct hypotheses.
DST_LABEL_MATCH_MIN_OVERLAP = 0.6
DST_CONFLICT_ESCALATE = 0.6
DST_IGNORANCE_RETRY = 0.5
# A hypothesis nothing supports but nothing excludes is what has not been ruled out.
# Below this plausibility it is not worth raising with the patient.
DST_UNRULED_MIN_PLAUSIBILITY = 0.3
# Whether the fusion may CHANGE the outcome, or only record what it would have done.
# Ships false deliberately: the frame is built by string-matching diagnosis labels, and
# until that is measured on real traffic a mismatch would fire the safety clamps for the
# wrong reason. Measure first, then enforce.
DST_ENFORCE = False

# Per-turn traces — one JSON line per turn, for offline divergence analysis between the
# orchestrator's synthesis and the fusion. Off the request path's critical logic: a
# tracing failure must never affect a turn.
TRACE_ENABLED = True
TRACE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "traces", "turns.jsonl"
)

# Embedding batch size
EMBEDDING_BATCH_SIZE = 100

# Pinecone upsert batch size
PINECONE_UPSERT_BATCH_SIZE = 100

# Max tokens for embedding model (text-embedding-3-small limit)
EMBEDDING_MAX_TOKENS = 8191