"""
app/main.py
------------
FastAPI application factory.

Startup:
  - Loads the FAISS vector store from disk (built by scripts/ingest.py).
  - Mounts the API router.

Run:
  uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.services.retrieval_service import load_vector_store, vector_store_exists

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Campus Help Assistant",
    description=(
        "A backend-only mini RAG chatbot that answers student questions "
        "from the Campus Handbook."
    ),
    version="1.0.0",
)

app.include_router(router)


# ── Startup event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    """Load the FAISS vector store once when the server starts."""
    if not vector_store_exists(settings.VECTOR_STORE_DIR):
        logger.warning(
            "Vector store not found at '%s'. "
            "Run `python scripts/ingest.py` before sending chat requests.",
            settings.VECTOR_STORE_DIR,
        )
        return

    load_vector_store(settings.VECTOR_STORE_DIR)
    logger.info("Campus Help Assistant is ready.")
