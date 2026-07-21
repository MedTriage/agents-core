"""SNOMED CT knowledge graph, backed by SQLite.

This is the ONLY module that knows how the graph is stored. `kgrag_node` talks to the
five functions at the bottom and never sees SQL — so swapping SQLite for Neo4j later
means rewriting this file alone.

Why SQLite rather than Neo4j: the KGRAG branch performs exactly four operations —
term→concept, concept→ancestors, concept→descendants, concept→attributes. That is one
full-text lookup (FTS5, the same class of inverted index as Neo4j's Lucene) and three
depth-bounded hierarchy walks (recursive CTEs). No server, no JVM, no size cap, and the
full ~370k-concept release fits comfortably.
"""
import os
import re
import sqlite3

from app.config import (
    SNOMED_DB_PATH,
    SNOMED_MAX_CONCEPTS,
    SNOMED_MAX_CONCEPTS_PER_TERM,
    SNOMED_MAX_DEPTH,
    SNOMED_MAX_RELATIVES,
)

# SNOMED CT metadata concept ids. These are stable, part of the standard, and used to
# pick the relationships worth traversing out of the ~1.2M in a release.
IS_A = 116680003
FULLY_SPECIFIED_NAME = 900000000000003001

# The attribute relationships that actually carry clinical meaning for triage. SNOMED
# defines many more; these are the ones that explain *what* a disorder is about.
ATTRIBUTE_NAMES = {
    363698007: "Finding site",
    116676008: "Associated morphology",
    246075003: "Causative agent",
    42752001: "Due to",
    255234002: "After",
    263502005: "Clinical course",
    246112005: "Severity",
    370135005: "Pathological process",
    246454002: "Occurrence",
    47429007: "Associated with",
    418775008: "Finding method",
    260686004: "Method",
}


# Every FSN ends in a semantic tag: "Diphtheria (disorder)". Only these three describe
# something the patient can HAVE. The one that matters is the tag we exclude:
# "(situation)" wraps context around a finding, including its negation — "No sore
# throat (situation)" is a genuine concept and a genuine lexical match for the words a
# patient used, and linking to it inverts what they said.
ALLOWED_SEMANTIC_TAGS = {"finding", "disorder", "event"}

_SEMANTIC_TAG = re.compile(r"\(([^)]*)\)\s*$")

# Rows to pull per term before filtering. The tag filter rejects a good fraction of
# lexical matches, so the SQL has to over-fetch to still fill the per-term budget.
_TAG_FILTER_OVERSAMPLE = 5


class SnomedUnavailable(RuntimeError):
    """The SNOMED graph has not been built. The KGRAG branch cannot run."""


def _connect() -> sqlite3.Connection:
    if not os.path.exists(SNOMED_DB_PATH):
        raise SnomedUnavailable(
            f"SNOMED graph not found at {SNOMED_DB_PATH}. "
            "Build it with: python -m app.rag.snomed_ingest <path-to-RF2-Snapshot>"
        )
    conn = sqlite3.connect(f"file:{SNOMED_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _fts_query(term: str) -> str:
    """Turn a free-text clinical phrase into an FTS5 AND-query.

    Every word must appear in the description, which is what makes "sore throat" match
    "Sore throat (finding)" but not every description containing merely "throat".
    Punctuation is stripped because FTS5 treats bare `-`, `"` etc. as operators and
    would otherwise raise a syntax error on ordinary clinical phrasing.
    """
    words = [w for w in "".join(c if c.isalnum() else " " for c in term).split() if w]
    if not words:
        return ""
    return " AND ".join(f'"{w}"' for w in words)


def _semantic_tag(fsn: str) -> str:
    match = _SEMANTIC_TAG.search(fsn or "")
    return match.group(1).strip().lower() if match else ""


def find_concepts(
    terms: list[str],
    limit: int = SNOMED_MAX_CONCEPTS,
    per_term: int = SNOMED_MAX_CONCEPTS_PER_TERM,
) -> list[dict]:
    """Map free-text clinical terms onto SNOMED concepts (the entity-linking step).

    SNOMED's own Description records are the lexicon — no external NER model. Ranked by
    FTS5 relevance, with shorter descriptions preferred on ties: "Sore throat" should
    win over "Sore throat co-occurrent with acute pharyngitis" for the bare phrase.

    Each term gets its own small budget and the results are then interleaved, so every
    term is represented before any term takes a second slot. This matters more than it
    looks: bm25 scores a one-word query like "fever" highly against every short
    description containing it, so without a per-term cap that single term returns Q
    fever, Piry virus disease and malt-workers' lung, exhausts the budget, and the
    patient's actual complaint never gets linked.
    """
    conn = _connect()
    try:
        per_term_hits: list[list[dict]] = []
        seen: set[int] = set()

        for term in terms:
            query = _fts_query(term)
            if not query:
                continue

            rows = conn.execute(
                """
                SELECT c.id      AS id,
                       c.fsn     AS fsn,
                       d.term    AS matched_term,
                       bm25(description_fts) AS rank
                FROM description_fts
                JOIN description d ON d.rowid = description_fts.rowid
                JOIN concept     c ON c.id = d.concept_id
                WHERE description_fts MATCH ?
                  AND c.active = 1
                  AND d.active = 1
                ORDER BY rank, length(d.term)
                LIMIT ?
                """,
                (query, per_term * _TAG_FILTER_OVERSAMPLE),
            ).fetchall()

            hits: list[dict] = []
            for row in rows:
                # First match for a concept wins — rows are already relevance-ordered.
                if row["id"] in seen:
                    continue
                if _semantic_tag(row["fsn"]) not in ALLOWED_SEMANTIC_TAGS:
                    continue

                seen.add(row["id"])
                hits.append(
                    {
                        "id": row["id"],
                        "fsn": row["fsn"],
                        "matched_term": row["matched_term"],
                        "matched_query": term,
                        # FTS5 bm25: more negative is a better match. Surfaced raw —
                        # callers that need a match-quality measure normalise it.
                        "rank": row["rank"],
                    }
                )
                if len(hits) >= per_term:
                    break

            per_term_hits.append(hits)

        # Round-robin: every term's best match, then every term's second, and so on.
        ordered = [
            hits[i]
            for i in range(per_term)
            for hits in per_term_hits
            if i < len(hits)
        ]
        return ordered[:limit]
    finally:
        conn.close()


def get_ancestors(
    concept_id: int,
    max_depth: int = SNOMED_MAX_DEPTH,
    limit: int = SNOMED_MAX_RELATIVES,
) -> list[dict]:
    """Walk IS_A upward: what more general conditions is this a kind of?"""
    return _walk(concept_id, max_depth, limit, upward=True)


def get_descendants(
    concept_id: int,
    max_depth: int = SNOMED_MAX_DEPTH,
    limit: int = SNOMED_MAX_RELATIVES,
) -> list[dict]:
    """Walk IS_A downward: what more specific conditions fall under this?

    This is where KGRAG earns its keep for differentials — the children of a general
    finding are exactly the specific disorders worth ruling out.

    Bounded hard: a general concept ("pain") has thousands of descendants in the real
    release. Nearest-first, so the closest — and most clinically relevant — survive
    the cut.
    """
    return _walk(concept_id, max_depth, limit, upward=False)


def _walk(concept_id: int, max_depth: int, limit: int, upward: bool) -> list[dict]:
    # Direction is the only thing that differs between the two traversals.
    from_col, to_col = ("source_id", "destination_id") if upward else ("destination_id", "source_id")

    conn = _connect()
    try:
        rows = conn.execute(
            f"""
            WITH RECURSIVE walk(id, depth) AS (
                SELECT ?, 0
                UNION
                SELECT r.{to_col}, w.depth + 1
                FROM relationship r
                JOIN walk w ON r.{from_col} = w.id
                WHERE r.type_id = {IS_A}
                  AND r.active = 1
                  AND w.depth < ?
            )
            SELECT c.id AS id, c.fsn AS fsn, w.depth AS depth
            FROM walk w
            JOIN concept c ON c.id = w.id
            WHERE w.depth > 0
              AND c.active = 1
            ORDER BY w.depth, c.fsn
            LIMIT ?
            """,
            (concept_id, max_depth, limit),
        ).fetchall()

        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_attributes(concept_id: int) -> list[dict]:
    """The defining attributes of a concept — finding site, causative agent, morphology.

    This is the part a vector store cannot give you: SNOMED states outright that
    diphtheria is *caused by* Corynebacterium diphtheriae and *sited in* the pharynx.
    """
    conn = _connect()
    try:
        placeholders = ",".join("?" * len(ATTRIBUTE_NAMES))
        rows = conn.execute(
            f"""
            SELECT r.type_id AS type_id,
                   c.id      AS value_id,
                   c.fsn     AS value_fsn
            FROM relationship r
            JOIN concept c ON c.id = r.destination_id
            WHERE r.source_id = ?
              AND r.active = 1
              AND c.active = 1
              AND r.type_id IN ({placeholders})
            ORDER BY r.type_id
            """,
            (concept_id, *ATTRIBUTE_NAMES.keys()),
        ).fetchall()

        return [
            {
                "relationship": ATTRIBUTE_NAMES[r["type_id"]],
                "value_id": r["value_id"],
                "value_fsn": r["value_fsn"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def is_available() -> bool:
    return os.path.exists(SNOMED_DB_PATH)


def stats() -> dict:
    conn = _connect()
    try:
        return {
            "concepts": conn.execute("SELECT count(*) FROM concept WHERE active = 1").fetchone()[0],
            "descriptions": conn.execute("SELECT count(*) FROM description WHERE active = 1").fetchone()[0],
            "relationships": conn.execute("SELECT count(*) FROM relationship WHERE active = 1").fetchone()[0],
        }
    finally:
        conn.close()
