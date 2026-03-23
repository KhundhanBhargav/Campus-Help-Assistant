"""
app/api/routes.py
------------------
Steps 17, 18, 19  –  FastAPI route definitions.

Endpoints:
  GET  /health   – liveness probe
  POST /chat     – main RAG chatbot endpoint
"""

import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.models import ChatRequest, ChatResponse, HealthResponse, RetrievedSource
from app.services.llm_service import call_llm
from app.services.prompt_service import NO_ANSWER_RESPONSE, build_prompt
from app.services.retrieval_service import retrieve_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Simple liveness probe — returns {"status": "ok"}."""
    return HealthResponse(status="ok")


# ── POST /chat ────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Main RAG endpoint.

    1. Validate & preprocess the question.
    2. Retrieve top-k chunks via semantic search.
    3. Build grounded prompt.
    4. Call the LLM.
    5. Return answer + sources.
    """
    question = request.question.strip()

    # Step 11 – validate input (Pydantic already rejects empty string; belt-and-suspenders)
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    logger.info("Incoming question: %s", question)

    # Steps 12 & 13 – embed query and retrieve chunks
    retrieved = retrieve_chunks(
        question=question,
        model_name=settings.EMBEDDING_MODEL,
        top_k=settings.TOP_K,
    )

    chunk_ids = [m["chunk_id"] for m, _ in retrieved]
    section_titles = [m["section_title"] for m, _ in retrieved]
    logger.info("Retrieved chunk_ids: %s", chunk_ids)
    logger.info("Retrieved sections: %s", section_titles)

    # Step 15 – build prompt
    system_prompt, user_message = build_prompt(question, retrieved)

    # Step 16 – call LLM
    answer = call_llm(system_prompt, user_message)

    # Step 17 – format response
    # Only include sources whose content was actually passed to the LLM
    from app.services.prompt_service import is_relevant
    sources = [
        RetrievedSource(
            chunk_id=meta["chunk_id"],
            section_title=meta["section_title"],
            source_file=meta["source_file"],
        )
        for meta, dist in retrieved
        if is_relevant(dist)
    ]

    logger.info("Answer length: %d chars | sources: %s", len(answer), [s.chunk_id for s in sources])

    return ChatResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks_count=len(retrieved),
    )
