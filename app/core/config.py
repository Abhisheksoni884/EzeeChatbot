"""
Core configuration — reads from environment variables.
Set GROQ_API_KEY in your .env file before running.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Groq — open-source Llama model via Groq inference API
    GROQ_API_KEY: str = ""
    # llama-3.1-8b-instant: fast, free-tier open-source model on Groq
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # Embeddings — open-source, runs locally, no API key needed
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG retrieval
    RETRIEVAL_TOP_K: int = 4          # number of chunks to retrieve per query
    CHUNK_SIZE: int = 512             # target characters per chunk
    CHUNK_OVERLAP: int = 64           # overlap between consecutive chunks

    # Storage path for FAISS indexes (one sub-dir per bot_id)
    VECTOR_STORE_DIR: str = "data/vector_stores"

    # Token cost estimate for llama-3.1-8b-instant on Groq (USD per 1M tokens)
    # Source: Groq pricing page (approximate; update as needed)
    COST_PER_1M_INPUT_TOKENS: float = 0.05
    COST_PER_1M_OUTPUT_TOKENS: float = 0.08

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
