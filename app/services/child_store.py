"""The child-health document store: FAISS over local embeddings.

The only module that knows how the documents are stored, as snomed_store.py is for SNOMED.
Exact inner-product search over normalised vectors (cosine similarity): the corpus is a few
documents, so an approximate index would buy nothing.
"""
import json
import threading
from pathlib import Path

import faiss
import numpy as np

from app.config import CHILD_EMBEDDING_MODEL, CHILD_INDEX_DIR, CHILD_MIN_SCORE, CHILD_TOP_K


class ChildIndexMissing(Exception):
    pass


_model = None
_loaded: dict[str, tuple] = {}
_lock = threading.Lock()


def _embedder():
    global _model
    if _model is None:
        from fastembed import TextEmbedding  # loads ONNX Runtime; only when first needed
        _model = TextEmbedding(CHILD_EMBEDDING_MODEL)
    return _model


def _embed_passages(texts: list[str]) -> np.ndarray:
    return np.array(list(_embedder().passage_embed(texts)), dtype="float32")


def _embed_query(text: str) -> np.ndarray:
    # bge models expect queries and passages to be embedded differently.
    return np.array(next(iter(_embedder().query_embed(text))), dtype="float32")


def build(chunks: list[dict], index_dir) -> None:
    """Embed `chunks` ({id, source, page, text}) and write the index beside them."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectors = _embed_passages([c["text"] for c in chunks])
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "chunks.json").write_text(json.dumps(chunks))
    with _lock:
        _loaded.pop(str(index_dir), None)


def _load(index_dir: Path):
    key = str(index_dir)
    with _lock:
        if key not in _loaded:
            if not (index_dir / "index.faiss").exists():
                raise ChildIndexMissing(f"No child-health index at {index_dir}; run app.rag.child_ingest.")
            _loaded[key] = (faiss.read_index(str(index_dir / "index.faiss")),
                            json.loads((index_dir / "chunks.json").read_text()))
        return _loaded[key]


def search(query: str, k: int = CHILD_TOP_K, min_score: float = CHILD_MIN_SCORE,
           index_dir=CHILD_INDEX_DIR) -> list[dict]:
    """The `k` passages nearest `query`, best first, each with its cosine `score`; passages
    below `min_score` are left out."""
    index, chunks = _load(Path(index_dir))
    vector = _embed_query(query).reshape(1, -1)
    faiss.normalize_L2(vector)
    scores, ids = index.search(vector, min(k, index.ntotal))
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])
            if i >= 0 and s >= min_score]
