"""
Vector store service — FAISS + HuggingFace embeddings.

Each bot gets its own FAISS index stored in a subdirectory under
VECTOR_STORE_DIR/<bot_id>/.  This ensures complete multi-bot isolation:
one bot cannot retrieve chunks from another bot's knowledge base.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  - Fully open-source, runs locally (no API key needed)
  - 384-dimensional vectors, fast and accurate for semantic search
  - Widely used for production RAG systems
"""

import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.core.config import settings


# ─── Embeddings (module-level singleton — loaded once, reused) ────────────────
# Loading the model is ~2s; keeping it as a module-level object avoids
# reloading it on every upload/chat request.
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            # encode_kwargs controls the embedding computation
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ─── Path helpers ─────────────────────────────────────────────────────────────

def _index_path(bot_id: str) -> str:
    """Return the directory where this bot's FAISS index is stored."""
    return os.path.join(settings.VECTOR_STORE_DIR, bot_id)


# ─── Public API ───────────────────────────────────────────────────────────────

def build_and_save_index(bot_id: str, chunks: List[Document]) -> None:
    """
    Embed all chunks and persist the FAISS index to disk.

    Args:
        bot_id: unique identifier for this bot's knowledge base
        chunks: list of Document chunks from the splitter
    """
    embeddings = _get_embeddings()
    index_dir = _index_path(bot_id)
    os.makedirs(index_dir, exist_ok=True)

    vector_store = FAISS.from_documents(chunks, embeddings)
    # allow_dangerous_deserialization is required by LangChain when saving
    # locally; safe here because WE write the file.
    vector_store.save_local(index_dir)


def load_index(bot_id: str) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Raises:
        FileNotFoundError: if no index exists for the given bot_id
    """
    index_dir = _index_path(bot_id)
    if not os.path.exists(index_dir):
        raise FileNotFoundError(
            f"No knowledge base found for bot_id='{bot_id}'. "
            "Please upload content first via POST /upload."
        )
    embeddings = _get_embeddings()
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we wrote this file
    )


def retrieve_chunks(bot_id: str, query: str) -> List[Document]:
    """
    Find the most relevant chunks for a query using similarity search.

    Args:
        bot_id: which bot's knowledge base to search
        query: the user's question (embedded on the fly)

    Returns:
        Top-K Document chunks, ordered by relevance score
    """
    vector_store = load_index(bot_id)
    # similarity_search returns top-K docs without exposing raw scores;
    # that's fine for RAG — we just need the content.
    return vector_store.similarity_search(query, k=settings.RETRIEVAL_TOP_K)


def bot_index_exists(bot_id: str) -> bool:
    """Check whether a FAISS index exists for the given bot_id."""
    return os.path.exists(_index_path(bot_id))
