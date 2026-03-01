import os
import hashlib
from tqdm import tqdm

from app.rag.chunker import extract_text_from_pdf, chunk_text
from app.rag.embedder import embed_texts_batch
from app.services.pinecone_client import index
from app.config import EMBEDDING_BATCH_SIZE, PINECONE_UPSERT_BATCH_SIZE

DOCS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "docs")


# Pinecone metadata limit is 40KB. Reserve some space for source/chunk_index fields.
MAX_METADATA_TEXT_BYTES = 38_000


def _truncate_metadata_text(text: str, max_bytes: int = MAX_METADATA_TEXT_BYTES) -> str:
    """Truncate text to fit within Pinecone's metadata size limit."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()


def _deterministic_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic ID based on source filename and chunk index.
    Re-running ingest with the same docs overwrites instead of duplicating."""
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _batch_upsert(vectors: list[dict], batch_size: int = PINECONE_UPSERT_BATCH_SIZE):
    """Upsert vectors to Pinecone in batches."""
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(batch)


def ingest():
    if not os.path.exists(DOCS_PATH):
        print(f"Docs directory not found: {DOCS_PATH}")
        return

    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in docs directory.")
        return

    for filename in pdf_files:
        print(f"\nIngesting: {filename}")
        path = os.path.join(DOCS_PATH, filename)

        text = extract_text_from_pdf(path)
        if not text.strip():
            print(f"  Skipping {filename} — no text extracted.")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(f"  Skipping {filename} — no chunks produced.")
            continue

        print(f"  Extracted {len(chunks)} chunks. Embedding...")

        # Batch embed all chunks
        all_embeddings = []
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
            embeddings = embed_texts_batch(batch)
            all_embeddings.extend(embeddings)

        # Build vectors with deterministic IDs
        vectors = []
        for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, all_embeddings), total=len(chunks), desc="  Preparing vectors")):
            vectors.append({
                "id": _deterministic_id(filename, i),
                "values": embedding,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "text": _truncate_metadata_text(chunk)
                }
            })

        # Batch upsert to Pinecone
        print(f"  Upserting {len(vectors)} vectors...")
        _batch_upsert(vectors)
        print(f"  Done: {filename}")

    print("\nIngestion complete.")


if __name__ == "__main__":
    ingest()