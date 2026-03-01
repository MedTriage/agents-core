from app.rag.embedder import embed_text
from app.services.pinecone_client import index
from app.config import RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD


def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K, score_threshold: float = RETRIEVAL_SCORE_THRESHOLD) -> list[dict]:
    """
    Retrieve relevant documents from Pinecone for a given query.
    Filters results below the score threshold to avoid low-relevance noise.
    """
    try:
        query_embedding = embed_text(query)

        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        documents = []

        for match in results["matches"]:
            score = match["score"]

            # Skip low-relevance results
            if score < score_threshold:
                continue

            documents.append({
                "score": score,
                "source": match["metadata"].get("source", "unknown"),
                "text": match["metadata"].get("text", "")
            })

        return documents

    except Exception as e:
        print(f"Retrieval error: {e}")
        return []