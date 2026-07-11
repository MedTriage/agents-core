"""Load a SNOMED CT RF2 release into the SQLite knowledge graph.

Usage:
    python -m app.rag.snomed_ingest /path/to/SnomedCT_.../Snapshot

Point it at the **Snapshot** directory of an RF2 release (not Full, not Delta —
Snapshot is the current state of every component, which is what we want). It finds the
Concept / Description / Relationship files underneath, so either the Snapshot root or
its Terminology subdirectory works.

RF2 files are tab-delimited with a header row. Inactive rows are kept but flagged, so
the store can filter on `active` rather than us silently discarding history.
"""
import os
import sqlite3
import sys

from app.config import SNOMED_DB_PATH

SCHEMA = """
CREATE TABLE concept (
    id                   INTEGER PRIMARY KEY,
    active               INTEGER NOT NULL,
    definition_status_id INTEGER,
    fsn                  TEXT
);

CREATE TABLE description (
    id         INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    active     INTEGER NOT NULL,
    type_id    INTEGER,
    term       TEXT NOT NULL
);

CREATE TABLE relationship (
    id             INTEGER PRIMARY KEY,
    source_id      INTEGER NOT NULL,
    destination_id INTEGER NOT NULL,
    active         INTEGER NOT NULL,
    type_id        INTEGER NOT NULL
);
"""

# Built after the bulk load — writing to indexes during insert is far slower.
INDEXES = """
CREATE INDEX idx_desc_concept  ON description(concept_id);
CREATE INDEX idx_rel_source    ON relationship(source_id, type_id, active);
CREATE INDEX idx_rel_dest      ON relationship(destination_id, type_id, active);

CREATE VIRTUAL TABLE description_fts USING fts5(
    term,
    content='description',
    content_rowid='id',
    tokenize='porter unicode61'
);

INSERT INTO description_fts(rowid, term) SELECT id, term FROM description;
"""

FULLY_SPECIFIED_NAME = 900000000000003001


def _find_file(root: str, prefix: str) -> str:
    """Locate an RF2 file by prefix. Filenames carry a release date, so we match loosely."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith(prefix) and name.endswith(".txt"):
                return os.path.join(dirpath, name)
    raise FileNotFoundError(
        f"No RF2 file starting with {prefix!r} under {root}. "
        "Point this at the Snapshot directory of a SNOMED CT release."
    )


def _rows(path: str):
    """Yield RF2 rows as lists of fields, skipping the header."""
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line.split("\t")


def ingest(release_dir: str, db_path: str = SNOMED_DB_PATH):
    if not os.path.isdir(release_dir):
        raise NotADirectoryError(f"Not a directory: {release_dir}")

    concept_file = _find_file(release_dir, "sct2_Concept_Snapshot")
    description_file = _find_file(release_dir, "sct2_Description_Snapshot")
    relationship_file = _find_file(release_dir, "sct2_Relationship_Snapshot")

    print(f"Concepts:      {concept_file}")
    print(f"Descriptions:  {description_file}")
    print(f"Relationships: {relationship_file}")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)  # rebuild from scratch — RF2 Snapshot is the full state

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    # Bulk-load settings. This is a one-shot build of a read-only artifact, so durability
    # guarantees during the load buy us nothing and cost a lot of time.
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")

    # id, effectiveTime, active, moduleId, definitionStatusId
    print("\nLoading concepts...")
    conn.executemany(
        "INSERT OR REPLACE INTO concept(id, active, definition_status_id) VALUES (?, ?, ?)",
        ((int(r[0]), int(r[2]), int(r[4])) for r in _rows(concept_file)),
    )
    conn.commit()

    # id, effectiveTime, active, moduleId, conceptId, languageCode, typeId, term, caseSignificanceId
    print("Loading descriptions...")
    conn.executemany(
        "INSERT OR REPLACE INTO description(id, concept_id, active, type_id, term) VALUES (?, ?, ?, ?, ?)",
        ((int(r[0]), int(r[4]), int(r[2]), int(r[6]), r[7]) for r in _rows(description_file)),
    )
    conn.commit()

    # id, effectiveTime, active, moduleId, sourceId, destinationId, relationshipGroup, typeId, ...
    print("Loading relationships...")
    conn.executemany(
        "INSERT OR REPLACE INTO relationship(id, source_id, destination_id, active, type_id) VALUES (?, ?, ?, ?, ?)",
        ((int(r[0]), int(r[4]), int(r[5]), int(r[2]), int(r[7])) for r in _rows(relationship_file)),
    )
    conn.commit()

    # Denormalise the FSN onto the concept so every lookup doesn't need a join back
    # into 1.4M descriptions just to name what it found.
    print("Attaching fully specified names...")
    conn.execute(
        """
        UPDATE concept
        SET fsn = (
            SELECT d.term FROM description d
            WHERE d.concept_id = concept.id
              AND d.type_id = ?
              AND d.active = 1
            LIMIT 1
        )
        """,
        (FULLY_SPECIFIED_NAME,),
    )
    conn.commit()

    print("Building indexes and full-text lexicon...")
    conn.executescript(INDEXES)
    conn.commit()

    counts = {
        "concepts": conn.execute("SELECT count(*) FROM concept WHERE active = 1").fetchone()[0],
        "descriptions": conn.execute("SELECT count(*) FROM description WHERE active = 1").fetchone()[0],
        "relationships": conn.execute("SELECT count(*) FROM relationship WHERE active = 1").fetchone()[0],
    }
    conn.execute("VACUUM")
    conn.close()

    size_mb = os.path.getsize(db_path) / 1_000_000
    print(f"\nDone: {db_path} ({size_mb:.0f} MB)")
    print(f"  active concepts:      {counts['concepts']:,}")
    print(f"  active descriptions:  {counts['descriptions']:,}")
    print(f"  active relationships: {counts['relationships']:,}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    ingest(sys.argv[1])
