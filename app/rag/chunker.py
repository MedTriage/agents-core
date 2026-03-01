import re
from pypdf import PdfReader
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file, page by page."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex, preserving medical abbreviations."""
    # Split on sentence-ending punctuation followed by whitespace
    # Avoids splitting on common medical abbreviations like "e.g.", "i.e.", "Dr.", "mg."
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Sentence-aware chunking. Builds chunks by accumulating full sentences
    up to chunk_size characters. Overlap is achieved by carrying trailing
    sentences from the previous chunk into the next.
    """
    sentences = _split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If a single sentence exceeds chunk_size, split it into chunk_size pieces
        if sentence_length > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            # Hard-split the oversized sentence
            for start in range(0, sentence_length, chunk_size - overlap):
                sub = sentence[start:start + chunk_size]
                if sub.strip():
                    chunks.append(sub)
            continue

        # If adding this sentence exceeds the limit, flush and start overlap
        if current_length + sentence_length + 1 > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # Build overlap: take trailing sentences that fit within overlap size
            overlap_chunk = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) + 1 > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_length += len(s) + 1

            current_chunk = overlap_chunk
            current_length = sum(len(s) for s in current_chunk) + max(0, len(current_chunk) - 1)

        current_chunk.append(sentence)
        current_length += sentence_length + (1 if current_length > 0 else 0)

    # Flush remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Filter out empty or whitespace-only chunks
    return [c for c in chunks if c.strip()]