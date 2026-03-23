# Campus Help Assistant — Backend Mini RAG Chatbot

A small, fully-functional **Retrieval-Augmented Generation (RAG)** backend that answers student questions from the Campus Handbook.

---

## Project Structure

```
mini-rag-backend/
├── app/
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints (GET /health, POST /chat)
│   ├── services/
│   │   ├── ingestion_service.py # Read → chunk → save chunks.json
│   │   ├── retrieval_service.py # Embed chunks, build FAISS index, semantic search
│   │   ├── llm_service.py       # Call Anthropic Claude
│   │   └── prompt_service.py    # Build grounded prompt, relevance filter
│   ├── core/
│   │   ├── config.py            # Pydantic settings (loads .env)
│   │   └── models.py            # Request / response Pydantic models
│   └── main.py                  # FastAPI app + startup loader
│
├── data/
│   ├── raw/
│   │   └── campus_handbook.txt  # Source of truth (6 sections)
│   ├── processed/
│   │   └── chunks.json          # Auto-generated after ingestion
│   └── vector_store/            # Auto-generated FAISS index + metadata
│
├── scripts/
│   └── ingest.py                # One-shot pipeline: read → chunk → embed → store
│
├── tests/
│   └── test_chat_api.py         # Pytest tests (mocked, no API key needed)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
cd mini-rag-backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Open .env and set your Anthropic API key:
# LLM_API_KEY=sk-ant-...
```

### 3. Run Ingestion (one-time)

```bash
python scripts/ingest.py
```

This will:
- Read `data/raw/campus_handbook.txt`
- Split into 6 section chunks
- Save `data/processed/chunks.json`
- Generate embeddings (sentence-transformers)
- Save FAISS index to `data/vector_store/`

### 4. Start the Server

```bash
uvicorn app.main:app --reload
```

Server runs at **http://localhost:8000**

### 5. Try It

```bash
# Health check
curl http://localhost:8000/health

# Ask a supported question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the revaluation fee?"}'

# Ask an unsupported question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is the principal?"}'
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## API Reference

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{"question": "What is the hostel late fee?"}
```

**Response:**
```json
{
  "answer": "If the hostel fee is not paid within the 7-day grace period, a late fee of 200 rupees is charged. (Source: Hostel Fee Payment)",
  "sources": [
    {
      "chunk_id": "chunk_3",
      "section_title": "Hostel Fee Payment",
      "source_file": "campus_handbook.txt"
    }
  ],
  "retrieved_chunks_count": 3
}
```

**Unsupported question response:**
```json
{
  "answer": "I do not have enough information in the provided knowledge base to answer that.",
  "sources": [],
  "retrieved_chunks_count": 3
}
```

---

## Supported Questions

| Question | Source Section |
|---|---|
| What is the minimum attendance required? | Attendance Policy |
| How many books can a student borrow? | Library Rules |
| What is the library late fine? | Library Rules |
| What is the hostel grace period? | Hostel Fee Payment |
| What is the exam revaluation fee? | Examination Revaluation |
| How long does a duplicate ID card take? | ID Card Replacement |
| What GPA is needed for scholarship renewal? | Scholarship Renewal |

---

## RAG Flow (End-to-End)

```
User Question
     │
     ▼
Preprocess (strip whitespace, validate)
     │
     ▼
Embed Question  (sentence-transformers)
     │
     ▼
FAISS Semantic Search  →  Top-K Chunks
     │
     ▼
Relevance Filter  (distance threshold)
     │
  relevant?
  YES ──►  Build Prompt  →  Call Claude LLM  →  Answer + Sources
  NO  ──►  "I do not have enough information…"
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | *(required)* | Anthropic API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `CHAT_MODEL` | `claude-sonnet-4-20250514` | Anthropic chat model |
| `VECTOR_STORE_DIR` | `data/vector_store` | FAISS index location |
| `TOP_K` | `3` | Chunks retrieved per query |
