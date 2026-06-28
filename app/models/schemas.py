"""
Pydantic models for all API request and response bodies.
Kept in one place so changes to the contract are easy to find.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, HttpUrl


# ─── Upload ───────────────────────────────────────────────────────────────────

class UploadTextRequest(BaseModel):
    """Upload raw text as a knowledge base."""
    text: str


class UploadURLRequest(BaseModel):
    """Upload a URL; the API fetches and indexes its content."""
    url: HttpUrl


class UploadResponse(BaseModel):
    bot_id: str
    message: str
    chunks_indexed: int


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    bot_id: str
    user_message: str
    conversation_history: List[ConversationTurn] = []


# Chat responses stream as SSE, so there's no JSON response model for /chat.
# A non-streaming error still returns this:
class ChatErrorResponse(BaseModel):
    error: str


# ─── Stats ────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    bot_id: str
    total_messages: int
    avg_latency_ms: float
    estimated_cost_usd: float
    unanswered_questions: int
