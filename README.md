# EzeeChatBot

A minimal **RAG chatbot API** — upload a knowledge base, get a grounded chatbot.

Built with FastAPI, LangChain, FAISS, and [Groq](https://console.groq.com) (Llama 3.1 8B).

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## How it works

1. **Upload** plain text or a URL → content is chunked and indexed in a FAISS vector store
2. **Chat** — questions are answered strictly from the indexed content via streaming SSE
3. **Stats** — per-bot token usage, latency, and unanswered question metrics

Each upload returns a `bot_id`. All subsequent chat and stats requests use that ID — bots are fully isolated.

---

## Project structure

```
ezeechatbot/
├── main.py                  # FastAPI entry point
├── requirements.txt
├── .env.example
├── test_env.py              # Smoke-test: validates env, API key, and full RAG flow
└── app/
    ├── api/routes.py        # /upload  /chat  /stats
    ├── core/config.py       # Pydantic settings (reads .env)
    ├── models/schemas.py    # Request/response schemas
    └── services/
        ├── loader.py        # Text & URL ingestion
        ├── chunker.py       # RecursiveCharacterTextSplitter
        ├── vector_store.py  # FAISS + HuggingFace embeddings
        ├── llm.py           # ChatGroq streaming, prompt building, cost tracking
        └── stats_store.py   # Per-bot metrics
```

---

## Setup

**Prerequisites:** Python 3.11+, a free [Groq API key](https://console.groq.com)

```bash
git clone https://github.com/yourname/ezeechatbot.git
cd ezeechatbot

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set GROQ_API_KEY
```

**Verify everything works:**

```bash
python test_env.py
```

**Run the server:**

```bash
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## API

### `POST /upload`

Upload a knowledge base via `multipart/form-data`. Send exactly one of:

| Field | Type   | Description        |
|-------|--------|--------------------|
| `text` | string | Raw text content   |
| `url`  | string | Public URL to fetch |

```bash
curl -X POST http://localhost:8000/upload -F "text=Your knowledge base content here."
# or
curl -X POST http://localhost:8000/upload -F "url=https://example.com/article"
```

```json
{ "bot_id": "bot_a3f92c1d", "message": "Knowledge base indexed successfully.", "chunks_indexed": 14 }
```

---

### `POST /chat`

Ask a question. Response streams as SSE (`data: <token>\n\n`, ends with `data: [DONE]`).

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"bot_id": "bot_a3f92c1d", "user_message": "What is this about?", "conversation_history": []}' \
  --no-buffer
```

`conversation_history` is optional — pass `[]` for single-turn, or include prior `{role, content}` turns for multi-turn context.

If the answer isn't in the knowledge base, the bot responds with:
> *"I couldn't find that information in the uploaded knowledge base."*

---

### `GET /stats/{bot_id}`

```bash
curl http://localhost:8000/stats/bot_a3f92c1d
```

```json
{
  "bot_id": "bot_a3f92c1d",
  "total_messages": 42,
  "avg_latency_ms": 1234.56,
  "estimated_cost_usd": 0.00012345,
  "unanswered_questions": 3
}
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local HuggingFace embedding model |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per query |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server port |

---

## Design notes

**Embeddings run locally** — `all-MiniLM-L6-v2` requires no additional API key and produces 384-dimensional vectors suitable for production RAG workloads.

**Chunking** uses `RecursiveCharacterTextSplitter`, which prioritises paragraph → sentence → word boundaries before falling back to character splits. This keeps semantic units intact and improves retrieval accuracy.

**Hallucination prevention** is enforced at the prompt level — the system message strictly instructs the model to answer only from the provided context and to use the fallback phrase otherwise. The fallback rate is tracked in `/stats`.

---

## License

MIT
