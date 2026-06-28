"""
EzeeChatBot – RAG Chatbot API
Entry point: starts the FastAPI app.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title="EzeeChatBot API",
    description="A minimal RAG chatbot API — upload knowledge, ask questions.",
    version="1.0.0",
)

# Allow all origins for development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "EzeeChatBot API"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
