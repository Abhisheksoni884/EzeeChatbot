"""
API route handlers for EzeeChatBot.

Endpoints:
  POST /upload  — index a knowledge base (text or URL)
  POST /chat    — ask a question; streams response via SSE
  GET  /stats/{bot_id} — get per-bot metrics
"""

import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import HttpUrl

from app.models.schemas import (
    UploadResponse,
    ChatRequest,
    StatsResponse,
    ConversationTurn,
)
from app.services.loader import load_from_text, load_from_url
from app.services.chunker import split_document
from app.services.vector_store import build_and_save_index, retrieve_chunks, bot_index_exists
from app.services.llm import (
    stream_rag_response,
    is_unanswered,
    estimate_call_tokens,
    calculate_cost,
)
from app.services.stats_store import record_message, get_stats, init_bot_stats
from app.utils.helpers import generate_bot_id

router = APIRouter()


# ─── POST /upload ─────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, tags=["Knowledge Base"])
async def upload_knowledge(
    text: str | None = Form(default=None),
    url: str | None = Form(default=None),
):
    """
    Upload a knowledge base by providing either plain text or a URL.

    The API will:
    1. Fetch / accept the content
    2. Split it into semantic chunks
    3. Embed and store in a FAISS vector index
    4. Return a unique bot_id

    Send the request as multipart/form-data with ONE of:
      - text=<your text>
      - url=<https://example.com>
    """
    if not text and not url:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'text' or 'url' in the form body.",
        )
    if text and url:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of 'text' or 'url', not both.",
        )

    # --- Load content ---
    try:
        if text:
            document = load_from_text(text)
        else:
            document = load_from_url(str(url))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Chunk ---
    try:
        chunks = split_document(document)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Embed + index ---
    bot_id = generate_bot_id()
    try:
        build_and_save_index(bot_id, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {e}")

    # --- Init stats entry ---
    init_bot_stats(bot_id)

    return UploadResponse(
        bot_id=bot_id,
        message="Knowledge base indexed successfully.",
        chunks_indexed=len(chunks),
    )


# ─── POST /chat ───────────────────────────────────────────────────────────────

@router.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with a previously uploaded knowledge base.

    Streams the LLM response as Server-Sent Events (SSE).
    Each SSE event is: `data: <token>\\n\\n`
    The stream ends with: `data: [DONE]\\n\\n`

    The LLM is instructed to answer ONLY from the indexed knowledge.
    If no relevant information is found, it responds with a clear fallback message.
    """
    if not bot_index_exists(request.bot_id):
        raise HTTPException(
            status_code=404,
            detail=f"Bot '{request.bot_id}' not found. Upload a knowledge base first.",
        )

    # --- Retrieve relevant chunks ---
    try:
        chunks = retrieve_chunks(request.bot_id, request.user_message)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    # --- Stream generator ---
    async def token_stream() -> AsyncGenerator[str, None]:
        """
        Streams tokens and, after the stream completes, records stats.
        This runs inside the streaming response context.
        """
        start_time = time.monotonic()
        full_response = ""

        try:
            async for token in stream_rag_response(
                bot_id=request.bot_id,
                user_message=request.user_message,
                chunks=chunks,
                history=request.conversation_history,
            ):
                full_response += token
                # SSE format: each message is "data: <payload>\n\n"
                yield f"data: {token}\n\n"
        except Exception as e:
            # Surface errors to the client mid-stream
            yield f"data: [ERROR] {e}\n\n"
        finally:
            # Always record metrics, even on error
            latency_ms = (time.monotonic() - start_time) * 1000
            unanswered = is_unanswered(full_response)

            input_tokens, output_tokens = estimate_call_tokens(
                user_message=request.user_message,
                chunks=chunks,
                history=request.conversation_history,
                response_text=full_response,
            )
            cost = calculate_cost(input_tokens, output_tokens)

            record_message(
                bot_id=request.bot_id,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                was_unanswered=unanswered,
            )

            yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering for SSE
        },
    )


# ─── GET /stats/{bot_id} ──────────────────────────────────────────────────────

@router.get("/stats/{bot_id}", response_model=StatsResponse, tags=["Stats"])
async def get_bot_stats(bot_id: str):
    """
    Return usage statistics for a specific bot.

    Includes:
    - Total messages served
    - Average response latency (ms)
    - Estimated LLM token cost (USD)
    - Number of unanswered questions
    """
    stats = get_stats(bot_id)
    if stats is None:
        raise HTTPException(
            status_code=404,
            detail=f"No stats found for bot_id='{bot_id}'.",
        )

    total = stats["total_messages"]
    avg_latency = (
        stats["total_latency_ms"] / total if total > 0 else 0.0
    )
    estimated_cost = calculate_cost(
        stats["total_input_tokens"],
        stats["total_output_tokens"],
    )

    return StatsResponse(
        bot_id=bot_id,
        total_messages=total,
        avg_latency_ms=round(avg_latency, 2),
        estimated_cost_usd=estimated_cost,
        unanswered_questions=stats["unanswered_questions"],
    )
