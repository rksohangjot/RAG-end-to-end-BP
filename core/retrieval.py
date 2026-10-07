"""Retrieval: find the most relevant stored chunks for a question.

Embeds the question with the same model used to embed stored chunks, then
queries the vector store for the nearest matches, dropping any below
MIN_SCORE. A cross-encoder then re-scores a wider candidate pool against the
question and keeps the best top_k. No generation here.
"""

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from core.logger import logger
from core.vectorstore import embed_texts, get_collection

DEFAULT_TOP_K = 5
# Cosine similarity below this is treated as "not relevant". all-MiniLM-L6-v2
# scored ~0.37-0.76 on relevant question/chunk pairs and <=0.16 on unrelated
# ones in a small calibration; 0.25 sits in that gap. Heuristic — tune per corpus.
MIN_SCORE = 0.25

# Reranking: fetch RERANK_CANDIDATES by cosine, let a cross-encoder (reads the
# question and chunk together; slower but more accurate than comparing separate
# embeddings) re-order them, keep top_k. Cross-encoder scores are raw logits on a
# different scale, so MIN_SCORE is still applied to the cosine score beforehand.
USE_RERANKER = True
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float  # cosine similarity to the question
    rerank_score: float | None = None  # cross-encoder score, when reranking ran


def list_sources() -> dict[str, int]:
    """Map each source document in the store to its number of chunks."""
    metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for meta in metadatas:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return counts


def _rerank(question: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-order chunks by cross-encoder score and keep top_k; on failure keep cosine order."""
    try:
        scores = _get_reranker().predict([(question, c.text) for c in chunks])
    except Exception:
        logger.exception("Reranker failed; falling back to cosine order")
        return chunks[:top_k]
    for c, sc in zip(chunks, scores):
        c.rerank_score = float(sc)
    return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_SCORE,
    rerank: bool | None = None,
) -> list[RetrievedChunk]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    rerank = USE_RERANKER if rerank is None else rerank
    n_fetch = max(top_k, RERANK_CANDIDATES) if rerank else top_k

    query_embedding = embed_texts([question])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_fetch, collection.count()),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = [
        RetrievedChunk(
            text=doc,
            source=meta["source"],
            chunk_index=meta["chunk_index"],
            score=1 - dist,
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
    chunks = [c for c in chunks if c.score >= min_score]
    if rerank and len(chunks) > 1:
        return _rerank(question, chunks, top_k)
    return chunks[:top_k]
