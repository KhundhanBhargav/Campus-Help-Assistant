# Campus Help Assistant 🎓

A backend-only AI-powered chatbot built using **Retrieval-Augmented Generation (RAG)** architecture that answers student queries strictly from a predefined campus handbook.

---

## 📌 Project Description

Campus Help Assistant is a backend-only AI-powered chatbot built using Retrieval-Augmented Generation (RAG) architecture that answers student queries strictly from a predefined campus handbook. The system works by splitting the handbook into six sections, converting each section into vector embeddings using the sentence-transformers model, and storing them in a FAISS vector index for fast semantic search. When a student asks a question, the system embeds the query, retrieves the top three most relevant sections, and passes them as context to the Groq LLaMA language model to generate a grounded and factual answer. The chatbot prevents hallucination by strictly instructing the LLM to answer only from the retrieved context, and safely returns an insufficient information response for any question outside the knowledge base. Built with FastAPI, the project exposes clean REST API endpoints with structured JSON responses including source sections, and follows a modular service-based architecture that separates ingestion, retrieval, prompt building, and LLM integration into independent components.

---

## 🔄 RAG Pipeline

```
User Question
     │
     ▼
Embed Question (sentence-transformers)
     │
     ▼
Semantic Search → FAISS Vector Store → Top 3 Chunks
     │
     ▼
Relevance Check
     │
  Relevant?
  YES ──► Build Prompt → Groq LLaMA → Grounded Answer + Sources
  NO  ──► "I do not have enough information..."
```

---

## 🗂️ Project Structure

```
mini-rag-backend/
│
├── app/
│   ├── api/
│   │   └── routes.py               # API endpoints (/chat, /health)
│   ├── services/
│   │   ├── ingestion_service.py    # Reads and chunks the handbook
│   │   ├── retrieval_service.py    # Embeddings + FAISS search
│   │   ├── llm_service.py          # Calls Groq LLaMA API
│   │   └── prompt_service.py       # Builds grounded prompt
│   ├── core/
│   │   ├── config.py               # Centralized settings
│   │   └── models.py               # Request/Response data models
│   └── main.py                     # FastAPI app + startup
│
├── data/
│   ├── raw/
│   │   └── campus_handbook.txt     # Knowledge base (6 sections)
│   ├── processed/
│   │   └── chunks.json             # Auto-generated after ingestion
│   └── vector_store/               # Auto-generated FAISS index
│
├── scripts/
│   └── ingest.py                   # One-time ingestion pipeline
│
├── tests/
│   └── test_chat_api.py            # 5 automated tests
│
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (faiss-cpu) |
| LLM | Groq (llama-3.3-70b-versatile) |
| Validation | Pydantic v2 |
| Settings | pydantic-settings |
| Testing | Pytest |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mini-rag-backend.git
cd mini-rag-backend
```

### 2. Install Dependencies

```bash
py -m pip install -r requirements.txt
```

### 3. Get Free Groq API Key

👉 Go to **https://console.groq.com**
- Sign up with Google account
- Click **API Keys** → **Create API Key**
- Copy the key starting with `gsk_...`

### 4. Configure Environment

```bash
copy .env.example .env
```

Open `.env` and fill in your key:

```
LLM_API_KEY=gsk_your_groq_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHAT_MODEL=llama-3.3-70b-versatile
VECTOR_STORE_DIR=data/vector_store
TOP_K=3
```

### 5. Run Ingestion (One Time Only)

```bash
py scripts/ingest.py
```

This reads the handbook, creates chunks, generates embeddings, and builds the FAISS index.

### 6. Start the Server

```bash
py -m uvicorn app.main:app --reload
```

Server runs at **http://localhost:8000**

### 7. Test via Swagger UI

👉 Open **http://localhost:8000/docs** in your browser to test all endpoints interactively.

---

## 📡 API Endpoints

### `GET /health`
Check if server is running.

**Response:**
```json
{
  "status": "ok"
}
```

---

### `POST /chat`
Ask a question from the campus handbook.

**Request:**
```json
{
  "question": "What is the revaluation fee?"
}
```

**Response:**
```json
{
  "answer": "The revaluation fee is 500 rupees per subject. Students can apply within 5 working days from the date of result publication.",
  "sources": [
    {
      "chunk_id": "chunk_4",
      "section_title": "Examination Revaluation",
      "source_file": "campus_handbook.txt"
    }
  ],
  "retrieved_chunks_count": 3
}
```

---

## ✅ Supported Questions

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

## ❌ Unsupported Questions

For questions outside the knowledge base, the chatbot safely responds:

```json
{
  "answer": "I do not have enough information in the provided knowledge base to answer that.",
  "sources": [],
  "retrieved_chunks_count": 3
}
```

Examples of unsupported questions:
- Who is the principal?
- What is the canteen menu?
- What are the bus timings?

---

## 🧪 Run Tests

Open a new terminal and run:

```bash
py -m pytest tests/ -v
```

**Expected output:**
```
tests/test_chat_api.py::test_health_endpoint                         PASSED
tests/test_chat_api.py::test_chat_valid_request                      PASSED
tests/test_chat_api.py::test_chat_empty_question                     PASSED
tests/test_chat_api.py::test_known_question_returns_grounded_answer  PASSED
tests/test_chat_api.py::test_unknown_question_returns_no_answer      PASSED

5 passed in 3.21s
```

> Tests use mocks — no API key or running server needed.

---

## 📝 Knowledge Base Sections

The chatbot answers only from these 6 sections:

1. **Attendance Policy** — Minimum 75% attendance required
2. **Library Rules** — Borrow up to 3 books, 5 rupees/day late fine
3. **Hostel Fee Payment** — 7 day grace period, 200 rupees late fee
4. **Examination Revaluation** — 500 rupees fee, apply within 5 days
5. **ID Card Replacement** — 300 rupees fee, issued in 3 working days
6. **Scholarship Renewal** — Minimum 8.0 GPA required

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | required | Groq API key |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence-transformers model |
| `CHAT_MODEL` | llama-3.3-70b-versatile | Groq LLM model |
| `VECTOR_STORE_DIR` | data/vector_store | FAISS index location |
| `TOP_K` | 3 | Number of chunks retrieved per query |

---

## 👨‍💻 Author

Built as a Backend Mini RAG Chatbot assignment demonstrating end-to-end RAG pipeline implementation with FastAPI, FAISS, and Groq LLaMA.
