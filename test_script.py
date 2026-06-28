"""
test_env.py  —  EzeeChatBot environment & integration test
===========================================================
Runs a series of checks in order:

  1. Config / .env loading
  2. Groq API key validity (live ping)
  3. Embedding model (local, no API key)
  4. End-to-end RAG flow  (upload text → chunk → embed → index → retrieve)
  5. LLM streaming smoke-test (one round-trip with the real Groq API)
  6. Stats store smoke-test

Run with:
    python test_env.py
"""

import asyncio
import sys
import textwrap

# ── colour helpers ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}v{RESET}  {msg}")
def fail(msg): print(f"  {RED}x{RESET}  {msg}"); sys.exit(1)
def info(msg): print(f"  {CYAN}i{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET}  {msg}")
def section(title): print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}\n{BOLD}  {title}{RESET}")


# ==============================================================================
# 1.  Config / .env loading
# ==============================================================================
section("1 - Config / .env loading")

try:
    from app.core.config import settings
    ok("app.core.config imported successfully")
except Exception as e:
    fail(f"Cannot import config: {e}")

checks = [
    ("GROQ_API_KEY",      settings.GROQ_API_KEY,      lambda v: v and len(v) > 10),
    ("GROQ_MODEL",        settings.GROQ_MODEL,        lambda v: v),
    ("EMBEDDING_MODEL",   settings.EMBEDDING_MODEL,   lambda v: v),
    ("RETRIEVAL_TOP_K",   settings.RETRIEVAL_TOP_K,   lambda v: isinstance(v, int) and v > 0),
    ("CHUNK_SIZE",        settings.CHUNK_SIZE,        lambda v: isinstance(v, int) and v > 0),
    ("CHUNK_OVERLAP",     settings.CHUNK_OVERLAP,     lambda v: isinstance(v, int) and v >= 0),
    ("HOST",              settings.HOST,              lambda v: v),
    ("PORT",              settings.PORT,              lambda v: isinstance(v, int) and v > 0),
    ("VECTOR_STORE_DIR",  settings.VECTOR_STORE_DIR,  lambda v: v),
]

all_ok = True
for name, value, predicate in checks:
    display = str(value) if name != "GROQ_API_KEY" else f"{str(value)[:8]}..."
    if predicate(value):
        ok(f"{name} = {display}")
    else:
        warn(f"{name} is missing or invalid ({display!r})")
        all_ok = False

if not all_ok:
    fail("One or more environment variables are not set correctly - fix your .env file.")


# ==============================================================================
# 2.  Groq API key validity (live ping)
# ==============================================================================
section("2 - Groq API key - live ping")

try:
    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": "Say: pong"}],
        max_tokens=5,
    )
    reply = response.choices[0].message.content.strip()
    ok(f"Groq API responded: {reply!r}")
    ok(f"Model in use: {settings.GROQ_MODEL}")
except Exception as e:
    fail(f"Groq API call failed: {e}")


# ==============================================================================
# 3.  Embedding model (local, no API key)
# ==============================================================================
section("3 - Embedding model (local, no API key)")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vector = embedder.embed_query("hello world")
    ok(f"Embedding model loaded: {settings.EMBEDDING_MODEL}")
    ok(f"Vector dimensions: {len(vector)}")
    if len(vector) > 0:
        ok("Embedding output is non-empty")
except Exception as e:
    fail(f"Embedding model failed: {e}")


# ==============================================================================
# 4.  End-to-end RAG flow
# ==============================================================================
section("4 - End-to-end RAG flow (upload -> chunk -> embed -> index -> retrieve)")

SAMPLE_TEXT = textwrap.dedent("""\
    EzeeChatBot is a minimal RAG (Retrieval-Augmented Generation) chatbot API
    built with FastAPI, LangChain, FAISS and the Groq inference API.

    It supports two knowledge-base upload modes:
      - Plain text  (POST /upload with form field 'text')
      - URL scraping (POST /upload with form field 'url')

    After uploading, the client receives a unique bot_id. That bot_id is used
    in subsequent POST /chat requests to retrieve relevant knowledge-base chunks
    and generate a grounded answer via the LLM.

    The embedding model (sentence-transformers/all-MiniLM-L6-v2) runs locally,
    so no extra API key is required for semantic search. Only the Groq API key
    (GROQ_API_KEY) is needed for the LLM generation step.

    Usage statistics - total messages, average latency, token cost, and
    unanswered-question count - are tracked per bot and exposed via
    GET /stats/{bot_id}.
""")

try:
    from app.services.loader import load_from_text
    doc = load_from_text(SAMPLE_TEXT)
    ok(f"Loader: created Document ({len(doc.page_content)} chars)")
except Exception as e:
    fail(f"Loader failed: {e}")

try:
    from app.services.chunker import split_document
    chunks = split_document(doc)
    ok(f"Chunker: produced {len(chunks)} chunk(s)")
    for i, c in enumerate(chunks, 1):
        info(f"  Chunk {i}: {len(c.page_content)} chars")
except Exception as e:
    fail(f"Chunker failed: {e}")

TEST_BOT_ID = "test-bot-env-check"

try:
    from app.services.vector_store import build_and_save_index, bot_index_exists, retrieve_chunks
    build_and_save_index(TEST_BOT_ID, chunks)
    ok(f"Vector store: FAISS index saved for bot_id='{TEST_BOT_ID}'")
except Exception as e:
    fail(f"Vector store build failed: {e}")

try:
    assert bot_index_exists(TEST_BOT_ID), "Index not found after saving"
    ok("Vector store: index existence check passed")
except Exception as e:
    fail(f"Index existence check failed: {e}")

try:
    results = retrieve_chunks(TEST_BOT_ID, "What embedding model is used?")
    ok(f"Retrieval: returned {len(results)} chunk(s) for test query")
    if results:
        snippet = results[0].page_content[:120].replace("\n", " ")
        info(f"  Top chunk preview: \"{snippet}...\"")
except Exception as e:
    fail(f"Retrieval failed: {e}")


# ==============================================================================
# 5.  LLM streaming smoke-test (real Groq round-trip)
# ==============================================================================
section("5 - LLM streaming smoke-test (real Groq round-trip)")

async def run_streaming_test():
    from app.services.llm import stream_rag_response
    retrieved = retrieve_chunks(TEST_BOT_ID, "What is EzeeChatBot?")
    tokens = []
    async for token in stream_rag_response(
        bot_id=TEST_BOT_ID,
        user_message="What is EzeeChatBot and what embedding model does it use?",
        chunks=retrieved,
        history=[],
    ):
        tokens.append(token)
    return "".join(tokens)

try:
    full_response = asyncio.run(run_streaming_test())
    ok(f"LLM stream completed - {len(full_response)} chars received")
    preview = full_response[:200].replace("\n", " ")
    info(f"  Response preview: \"{preview}...\"")
except Exception as e:
    fail(f"LLM streaming failed: {e}")


# ==============================================================================
# 6.  Stats store smoke-test
# ==============================================================================
section("6 - Stats store smoke-test")

try:
    from app.services.stats_store import init_bot_stats, record_message, get_stats
    init_bot_stats(TEST_BOT_ID)
    record_message(
        bot_id=TEST_BOT_ID,
        latency_ms=123.4,
        input_tokens=50,
        output_tokens=20,
        was_unanswered=False,
    )
    stats = get_stats(TEST_BOT_ID)
    assert stats is not None, "Stats returned None"
    assert stats["total_messages"] >= 1, "Message count not incremented"
    ok(f"Stats store: total_messages={stats['total_messages']}, "
       f"total_latency_ms={stats['total_latency_ms']:.1f}")
except Exception as e:
    fail(f"Stats store failed: {e}")


# ==============================================================================
# Summary
# ==============================================================================
print(f"\n{BOLD}{GREEN}{'=' * 60}")
print("  All checks passed - EzeeChatBot is fully operational!")
print(f"{'=' * 60}{RESET}\n")
