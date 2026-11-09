"""The child-health document store: FAISS over local embeddings, built from PDF pages. The
embedding model is replaced with a keyword-count fake, so no model is downloaded."""
import numpy as np
import pytest

from app.rag.child_ingest import chunks_from_pages
from app.services import child_store

VOCAB = ("fever", "diarrhoea", "cough", "rash")


def _fake_embed(texts):
    rows = [[text.lower().count(word) + 0.01 for word in VOCAB] for text in texts]
    return np.array(rows, dtype="float32")


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr(child_store, "_embed_passages", _fake_embed)
    monkeypatch.setattr(child_store, "_embed_query", lambda text: _fake_embed([text])[0])


CHUNKS = [
    {"id": "a", "source": "imci.pdf", "page": 4, "text": "Fever: give paracetamol for high fever."},
    {"id": "b", "source": "imci.pdf", "page": 9, "text": "Diarrhoea: give ORS after each loose stool. Diarrhoea"},
    {"id": "c", "source": "imnci.pdf", "page": 2, "text": "Cough or difficult breathing: count the breaths."},
]


def test_search_returns_the_nearest_passage_with_its_source_and_page(tmp_path):
    child_store.build(CHUNKS, tmp_path)

    found = child_store.search("my son has diarrhoea", k=1, min_score=0.0, index_dir=tmp_path)

    assert [(p["id"], p["source"], p["page"]) for p in found] == [("b", "imci.pdf", 9)]
    assert found[0]["text"].startswith("Diarrhoea") and 0 < found[0]["score"] <= 1.0001


def test_search_leaves_out_passages_below_the_minimum_score(tmp_path):
    child_store.build(CHUNKS, tmp_path)

    found = child_store.search("a rash", k=3, min_score=0.9, index_dir=tmp_path)

    assert found == []


def test_rebuilding_replaces_what_search_sees(tmp_path):
    child_store.build(CHUNKS, tmp_path)
    child_store.search("fever", index_dir=tmp_path, min_score=0.0)
    child_store.build(CHUNKS[2:], tmp_path)

    assert [p["id"] for p in child_store.search("fever", index_dir=tmp_path, min_score=0.0)] == ["c"]


def test_search_before_the_documents_are_ingested_says_so(tmp_path):
    with pytest.raises(child_store.ChildIndexMissing):
        child_store.search("fever", index_dir=tmp_path / "nothing-here")


def test_each_chunk_keeps_its_source_and_page_and_a_stable_id():
    pages = ["Fever in a young infant. Refer urgently.", "", "Give ORS. Continue breastfeeding."]

    chunks = chunks_from_pages("imci.pdf", pages)

    assert [(c["source"], c["page"]) for c in chunks] == [("imci.pdf", 1), ("imci.pdf", 3)]
    assert chunks == chunks_from_pages("imci.pdf", pages)  # same input, same ids
    assert len({c["id"] for c in chunks}) == len(chunks)
