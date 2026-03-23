"""
scripts/ingest.py
------------------
Step 10  –  One-shot ingestion pipeline.

Run once before starting the server:
    python scripts/ingest.py

What it does:
  1. Reads  data/raw/campus_handbook.txt
  2. Splits into section chunks
  3. Saves  data/processed/chunks.json
  4. Generates embeddings (sentence-transformers)
  5. Builds + saves FAISS index to data/vector_store/
"""

import logging
import sys
from pathlib import Path

# Make sure the project root is on sys.path so `app` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.services.ingestion_service import (
    load_raw_text,
    save_chunks,
    split_into_chunks,
)
from app.services.retrieval_service import build_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Campus Help Assistant – Ingestion Pipeline")
    logger.info("=" * 60)

    # ── Step 5: Load raw text ────────────────────────────────────────────────
    logger.info("[1/4] Reading knowledge base …")
    raw_text = load_raw_text(settings.RAW_DATA_PATH)

    # ── Step 6: Chunk by section ─────────────────────────────────────────────
    logger.info("[2/4] Splitting into chunks …")
    chunks = split_into_chunks(raw_text)
    for c in chunks:
        logger.info("       %s -> %s", c.chunk_id, c.section_title)

    # ── Step 7: Save chunks.json ─────────────────────────────────────────────
    logger.info("[3/4] Saving chunks.json …")
    save_chunks(chunks, settings.PROCESSED_CHUNKS_PATH)

    # ── Steps 8 & 9: Embed + store ───────────────────────────────────────────
    logger.info("[4/4] Generating embeddings and building vector store …")
    build_vector_store(
        chunks=chunks,
        store_dir=settings.VECTOR_STORE_DIR,
        model_name=settings.EMBEDDING_MODEL,
    )

    logger.info("=" * 60)
    logger.info("Ingestion complete. Start the server with:")
    logger.info("  uvicorn app.main:app --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
