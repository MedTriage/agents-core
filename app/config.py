# Configurations for handling environment variables and other settings for the application
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# All node generation goes through OpenRouter's OpenAI-compatible endpoint. OpenRouter
# namespaces model ids by upstream provider, hence the "openai/" prefix.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "openai/gpt-5.4-mini"
# One model id can be served by several upstream providers (OpenAI, Azure, ...), and
# OpenRouter is free to switch between them per request, which changes answers between
# otherwise identical calls. Routing is pinned and fallbacks are off, so a routing problem
# surfaces as an error rather than as silent drift.
OPENROUTER_PROVIDER = {"order": ["openai"], "allow_fallbacks": False}
# Decoding controls. Left unset, the API samples at temperature 1.0, and about a quarter
# of test cases flipped triage level between identical requests. Pinned so the same input
# gets the same triage. Reasoning models in the GPT-5 family reject an explicit
# temperature; the client drops the parameter on the first such rejection rather than
# failing. A gateway may also drop an unsupported parameter silently, so "pinned" is a
# statement about what was requested, not a guarantee about what the upstream applied.
OPENROUTER_TEMPERATURE = 0.0
OPENROUTER_SEED = 20260812

# Retry settings
MAX_RETRIES = 2

# Re-retrieval loop (orchestrator → retrieval branch retry). Applied per branch: the
# orchestrator may send each of kgrag/mcp back this many times independently.
#
# Zero as of the parametric pivot. The loop was built for the abstention cascade — every
# source declining and the turn falling through to a constrained no-diagnosis answer —
# and the parametric branch removes that failure by construction, since it always returns
# an assessment. A retry round costs up to five extra calls including a second
# orchestrator synthesis, the largest prompt in the turn, to re-ask sources that already
# answered.
#
# The machinery is left intact and config-driven rather than deleted: branch_issues,
# branch_refinement_hints and route_after_orchestrator all still work, and setting this
# back to 1 restores the loop.
MAX_BRANCH_RETRIES = 0

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
# Subtypes of a candidate disorder, and how many of them get enriched with their own
# defining attributes. Both are much tighter than under the old symptom-first traversal,
# where the descendants WERE the differential and 25 of them was the point. Hypothesis-
# first the candidate is already specific, its own defining attributes are the
# discriminator, and its subtypes are variants that mostly matter when they change
# management — at 25 enriched each they were 70% of a 27k-character synthesis prompt.
SNOMED_MAX_SUBTYPES = 6
SNOMED_MAX_ENRICHED_SUBTYPES = 3
# Ancestors shown per candidate. The walk is depth-bounded already and returns
# nearest-first, so the first few are the parents that say what kind of thing this is;
# past that it generalises into "clinical finding" and stops discriminating.
SNOMED_MAX_ANCESTORS_SHOWN = 6
# The patient's reported findings are linked too, but only to anchor the answer and the
# explainability graph — no hierarchy walk descends from them, since that is exactly the
# symptom-first traversal that produced findings instead of disorders. Kept small: they
# compete with the candidate disorders for prompt budget and carry less.
SNOMED_MAX_FINDING_CONCEPTS = 4
# Disorders sharing a candidate's immediate parent that nobody proposed — the ontology's
# own additions to the differential. Capped tightly: a common parent can have dozens of
# children and most are not clinically plausible for the presentation at hand.
SNOMED_MAX_SIBLINGS = 6

# --- Dempster-Shafer evidence fusion (app/services/dst.py) ---
# EVERY constant below is a CALIBRATION PARAMETER, not a tuned truth. They are
# placeholders pending a fit on a held-out query set.
#
# Evidence volume modulates an MCP result's strength rather than dominating it: a short
# but real result is still evidence. Floor is the volume multiplier at its minimum.
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
# Only the external evidence sources appear here: the parametric branch is excluded from
# the fusion entirely (see dst_node), so it has no discount to set.
DST_DISCOUNT = {"kgrag": 0.85, "mcp": 0.7}
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

# --- Accounts, persistence and email ---
# Postgres (see README for the one-line Podman command). Required: the API stores accounts,
# profiles, conversations and the review queue, and has nowhere else to put them.
DATABASE_URL = os.getenv("DATABASE_URL")
# Where links in emails point. Local by design: the system is not deployed.
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:3000")

# Sessions slide forward with use but never outlive the absolute cap.
SESSION_IDLE_DAYS = 14
SESSION_MAX_DAYS = 30
VERIFY_TOKEN_HOURS = 24
RESET_TOKEN_HOURS = 1
INVITE_TOKEN_DAYS = 7
PASSWORD_MIN_LENGTH = 10

# Login throttling: per email, and a looser per-IP ceiling for spraying across accounts.
LOGIN_MAX_FAILURES = 5
LOGIN_MAX_FAILURES_PER_IP = 20
LOGIN_FAILURE_WINDOW_MINUTES = 15

# A turn is ~9 paid LLM calls against an account capped at 20 requests a minute for every
# user combined, so one runaway client must not be able to drain either.
TURNS_PER_HOUR = 30
# How long a turn may hold its conversation before the busy flag is treated as abandoned.
TURN_BUSY_MINUTES = 10

# Doctors get at most one "cases waiting" email per window, not one per case.
DOCTOR_DIGEST_MINUTES = 30

# Gmail SMTP with an app password. Without credentials the outbox keeps queuing and
# sends nothing; no request ever fails because email is not configured.
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "MedTriage")
EMAIL_MAX_ATTEMPTS = 5
EMAIL_WORKER_ENABLED = os.getenv("EMAIL_WORKER_ENABLED", "1") == "1"
