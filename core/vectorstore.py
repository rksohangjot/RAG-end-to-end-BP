"""Vector store: embed chunks, persist them, and expose the collection.

Embeddings come from sentence-transformers (local, no API key). Storage is
Chroma, an embedded vector database that persists to a local folder, using
cosine similarity (matched to how sentence-transformers models are tuned).
core/retrieval.py queries the collection returned by get_collection().
Documents are deduplicated by a content hash stored in each chunk's metadata.
"""

from pathlib import Path
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer

from core.chunking import Chunk
from core.logger import logger

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "vector_store")
COLLECTION_NAME = "documents"

_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


class EmbeddingModelMismatch(RuntimeError):
    """The stored vectors were made by a different embedding model than the configured one."""


def _mismatch_message(detail: str) -> str:
    return (
        f"The knowledge base in '{PERSIST_DIR}' can't be used with the configured embedding "
        f"model '{EMBEDDING_MODEL_NAME}': {detail}. Mixing models would silently return "
        "wrong results. To rebuild: stop the app, delete that folder, and upload your "
        "documents again (or change EMBEDDING_MODEL_NAME back)."
    )


def _check_embedding_model(collection) -> None:
    """Make sure the store's vectors came from the configured model; stamp it if it's unmarked.

    Chroma ignores new metadata on an existing collection, and modify() replaces the
    metadata wholesale (the cosine setting lives in the collection's configuration,
    not its metadata, so only the new key is written).
    """
    stored = (collection.metadata or {}).get("embedding_model")
    if stored == EMBEDDING_MODEL_NAME:
        return
    if stored is not None:
        raise EmbeddingModelMismatch(_mismatch_message(f"it was built with '{stored}'"))

    # No model recorded: a store from before this check existed, or a brand-new empty one.
    if collection.count() > 0:
        stored_dim = len(collection.get(limit=1, include=["embeddings"])["embeddings"][0])
        model = _get_model()
        # Renamed in newer sentence-transformers; fall back for older installs.
        model_dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()
        if stored_dim != model_dim:
            raise EmbeddingModelMismatch(
                _mismatch_message(f"its vectors have {stored_dim} dimensions but the model produces {model_dim}")
            )
        logger.warning(
            "Vector store has no recorded embedding model; assuming it was built with %r "
            "(vector dimensions match)", EMBEDDING_MODEL_NAME
        )
    collection.modify(metadata={"embedding_model": EMBEDDING_MODEL_NAME})


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        collection = client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL_NAME},
        )
        _check_embedding_model(collection)
        _collection = collection  # only cached once the check passes, so a mismatch keeps failing
    return _collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return model.encode(texts).tolist()


def find_by_hash(content_hash: str) -> str | None:
    """Return the source name of a stored document with this content hash, if any."""
    found = get_collection().get(where={"content_hash": content_hash}, limit=1)
    metadatas = found["metadatas"]
    return metadatas[0]["source"] if metadatas else None


def ids_for_source(source: str) -> list[str]:
    """IDs of every stored chunk that came from the document with this name."""
    return get_collection().get(where={"source": source}, include=[])["ids"]


def delete_ids(ids: list[str]) -> None:
    if ids:
        get_collection().delete(ids=ids)


def add_chunks(chunks: list[Chunk], content_hash: str | None = None) -> int:
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    ids = [uuid4().hex for _ in chunks]
    metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]
    if content_hash:
        for meta in metadatas:
            meta["content_hash"] = content_hash

    collection = get_collection()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)
