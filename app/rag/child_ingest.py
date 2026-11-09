"""Build the child-health index from a folder of IMNCI/WHO PDFs.

    python -m app.rag.child_ingest data/child_docs

You supply the PDFs, after checking each one's copyright page: this repo neither downloads
nor redistributes them, and both the folder and the index are gitignored. Pages are read
one at a time so every passage keeps the page it came from, for the answer's citations.
"""
import hashlib
import sys
from pathlib import Path

from pypdf import PdfReader

from app.config import CHILD_INDEX_DIR
from app.rag.chunker import chunk_text
from app.services import child_store


def chunks_from_pages(source: str, pages: list[str]) -> list[dict]:
    """Chunk each page's text, keeping its 1-based page number. Ids are stable, so the same
    document always produces the same ids."""
    chunks = []
    for page, text in enumerate(pages, start=1):
        for i, piece in enumerate(chunk_text(text or "")):
            digest = hashlib.sha256(f"{source}:{page}:{i}".encode()).hexdigest()[:16]
            chunks.append({"id": digest, "source": source, "page": page, "text": piece})
    return chunks


def ingest(folder: Path, index_dir: Path = Path(CHILD_INDEX_DIR)) -> int:
    chunks = []
    for pdf in sorted(folder.glob("*.pdf")):
        pages = [page.extract_text() or "" for page in PdfReader(pdf).pages]
        found = chunks_from_pages(pdf.name, pages)
        print(f"{pdf.name}: {len(pages)} pages, {len(found)} chunks")
        chunks.extend(found)
    if not chunks:
        raise SystemExit(f"No text found in PDFs under {folder}")
    child_store.build(chunks, index_dir)
    print(f"Index written to {index_dir} ({len(chunks)} chunks)")
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m app.rag.child_ingest <folder of PDFs>")
    ingest(Path(sys.argv[1]))
