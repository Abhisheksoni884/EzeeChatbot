"""
LLM service — ChatGroq (llama-3.1-8b-instant) with streaming.

Responsibilities:
  1. Build a grounded RAG prompt from retrieved chunks + conversation history
  2. Stream the LLM response token-by-token (for /chat SSE endpoint)
  3. Detect "unanswered" responses (bot said it couldn't find the info)
  4. Estimate token counts for cost tracking

The system prompt explicitly instructs the model to:
  - ONLY use the provided context
  - Say the fallback phrase if the answer isn't in the context
  - Never make things up

This is how we prevent hallucination at the prompt level.
"""

from typing import AsyncIterator, List

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

from app.core.config import settings
from app.models.schemas import ConversationTurn


# Fallback phrase — any response containing this is flagged as "unanswered"
FALLBACK_PHRASE = "I couldn't find that information in the uploaded knowledge base."

# System prompt template — the {context} placeholder is filled at runtime
SYSTEM_PROMPT_TEMPLATE = """You are EzeeChatBot, a helpful assistant that answers questions \
strictly based on the provided knowledge base context.

RULES (follow exactly):
1. Answer ONLY using information from the CONTEXT below.
2. If the context does not contain enough information to answer the question, \
respond with exactly: "{fallback}"
3. Do NOT make up information, guess, or use knowledge outside the context.
4. Keep answers concise and accurate.
5. If the user's question is a follow-up, use the conversation history for \
pronouns and references, but still answer only from the context.

CONTEXT:
{context}
""".format(
    fallback=FALLBACK_PHRASE,
    context="{context}",  # stays as a runtime placeholder
)


def _build_context_string(chunks: List[Document]) -> str:
    """
    Concatenate retrieved chunks into a numbered context block.
    Numbering helps the model reference specific parts if needed.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Chunk {i}]\n{chunk.page_content.strip()}")
    return "\n\n".join(parts)


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token (standard heuristic).
    Not perfectly accurate, but sufficient for cost estimation.
    """
    return max(1, len(text) // 4)


def is_unanswered(response_text: str) -> bool:
    """Return True if the response contains the standard fallback phrase."""
    return FALLBACK_PHRASE.lower() in response_text.lower()


async def stream_rag_response(
    bot_id: str,
    user_message: str,
    chunks: List[Document],
    history: List[ConversationTurn],
) -> AsyncIterator[str]:
    """
    Build the RAG prompt and stream the LLM response token-by-token.

    Args:
        bot_id: for logging/context (not used in prompt but useful to keep)
        user_message: the current user question
        chunks: retrieved knowledge base chunks
        history: prior conversation turns (for multi-turn context)

    Yields:
        String tokens as they arrive from the Groq API
    """
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.1,      # low temperature = more grounded, less creative
        max_tokens=1024,
        streaming=True,
    )

    # --- Build the message list ---
    context_str = _build_context_string(chunks)
    system_content = SYSTEM_PROMPT_TEMPLATE.replace("{context}", context_str)

    messages = [SystemMessage(content=system_content)]

    # Add conversation history so the model handles follow-up questions
    for turn in history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))

    # Add the current user message
    messages.append(HumanMessage(content=user_message))

    # --- Stream ---
    async for chunk in llm.astream(messages):
        # chunk.content is the incremental token string
        if chunk.content:
            yield chunk.content


def estimate_call_tokens(
    user_message: str,
    chunks: List[Document],
    history: List[ConversationTurn],
    response_text: str,
) -> tuple[int, int]:
    """
    Estimate input and output token counts for cost tracking.

    Returns:
        (input_tokens, output_tokens)
    """
    context_str = _build_context_string(chunks)
    history_str = " ".join(t.content for t in history)

    input_text = SYSTEM_PROMPT_TEMPLATE + context_str + history_str + user_message
    input_tokens = _estimate_tokens(input_text)
    output_tokens = _estimate_tokens(response_text)

    return input_tokens, output_tokens


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Estimate USD cost based on Groq's per-token pricing.
    Rates are set in config.py and can be updated there.
    """
    input_cost = (input_tokens / 1_000_000) * settings.COST_PER_1M_INPUT_TOKENS
    output_cost = (output_tokens / 1_000_000) * settings.COST_PER_1M_OUTPUT_TOKENS
    return round(input_cost + output_cost, 8)
