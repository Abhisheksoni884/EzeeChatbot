"""
Chunking strategy — sentence-aware recursive splitting.

Why this approach?
------------------
Fixed-character chunking is simple but breaks sentences and paragraphs
mid-thought, which degrades retrieval quality because a retrieved chunk
may start or end with half a sentence, confusing the LLM.

We use LangChain's RecursiveCharacterTextSplitter which tries a
priority-ordered list of separators:

  1. Double newline  (\n\n) → preserves paragraph boundaries first
  2. Single newline  (\n)   → falls back to line breaks
  3. Sentence end    (. ! ?) → then sentence boundaries
  4. Space                   → word boundary as last resort
  5. Empty string            → character split only if all else fails

This gives chunks that are semantically coherent — they contain complete
sentences and respect paragraph structure — which improves both retrieval
precision and LLM answer quality.

Overlap (default 64 chars ≈ one short sentence) is added so that context
spanning a chunk boundary is not lost.

Each chunk is returned as a LangChain Document, ready for embedding.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


def split_document(document: Document) -> List[Document]:
    """
    Split a single Document into semantically coherent chunks.

    Args:
        document: LangChain Document (output of loader.py)

    Returns:
        List of Document chunks with inherited metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        # Ordered separators — the splitter tries each in sequence and
        # only moves to the next if a chunk would still exceed chunk_size.
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        # Keep separators so sentences don't lose their terminal punctuation
        keep_separator=True,
    )

    chunks = splitter.split_documents([document])

    if not chunks:
        raise RuntimeError("Document produced zero chunks after splitting.")

    return chunks
