"""
app/services/retrieval_service.py
-----------------------------------
Steps 8, 9, 11, 12, 13  –  Embed chunks, store in FAISS, retrieve by query.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.models import Chunk

logger = logging.getLogger(__name__)

# File names inside VECTOR_STORE_DIR
_INDEX_FILE = "faiss.index"
_META_FILE = "metadata.json"

# Module-level cache so the model & index are loaded only once per process
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_metadata: List[dict] | None = None


# ── Step 8: Embed text ───────────────────────────────────────────────────────

def _get_model(model_name: str) -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """Return a float32 numpy array of shape (N, dim)."""
    model = _get_model(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype("float32")


# ── Step 9: Build + save FAISS index ────────────────────────────────────────

def build_vector_store(
    chunks: List[Chunk],
    store_dir: str,
    model_name: str,
) -> None:
    """
    Embed every chunk and persist:
      - faiss.index  (FAISS flat-L2 index)
      - metadata.json (list of dicts with chunk info)
    """
    global _index, _metadata

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, model_name)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    store_path = Path(store_dir)
    store_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(store_path / _INDEX_FILE))

    meta = [
        {
            "chunk_id": c.chunk_id,
            "section_title": c.section_title,
            "text": c.text,
            "source_file": c.source_file,
        }
        for c in chunks
    ]
    with open(store_path / _META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Update in-memory cache
    _index = index
    _metadata = meta

    logger.info(
        "Vector store built: %d vectors (dim=%d) → %s", len(chunks), dim, store_dir
    )


# ── Load existing index ──────────────────────────────────────────────────────

def load_vector_store(store_dir: str) -> None:
    """Load FAISS index and metadata from disk into module-level cache."""
    global _index, _metadata

    store_path = Path(store_dir)
    index_path = store_path / _INDEX_FILE
    meta_path = store_path / _META_FILE

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {store_dir}. "
            "Run `python scripts/ingest.py` first."
        )

    _index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    logger.info("Vector store loaded: %d vectors from %s", _index.ntotal, store_dir)


# ── Step 11: Preprocess query ────────────────────────────────────────────────

def preprocess_query(question: str) -> str:
    """Strip whitespace; raise if empty."""
    cleaned = " ".join(question.split())
    if not cleaned:
        raise ValueError("Question must not be empty.")
    return cleaned


# ── Steps 12 & 13: Embed query + retrieve top-k chunks ───────────────────────

def retrieve_chunks(
    question: str,
    model_name: str,
    top_k: int = 3,
) -> List[Tuple[dict, float]]:
    """
    Embed the question and return a list of (metadata_dict, distance) tuples
    for the top_k most similar chunks.
    """
    if _index is None or _metadata is None:
        raise RuntimeError(
            "Vector store is not loaded. Call load_vector_store() at startup."
        )

    clean_q = preprocess_query(question)
    query_vec = embed_texts([clean_q], model_name)

    distances, indices = _index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:          # FAISS returns -1 when fewer results than top_k
            continue
        results.append((_metadata[idx], float(dist)))

    logger.info(
        "Retrieved %d chunks for query: '%s' | distances: %s",
        len(results),
        clean_q,
        [round(d, 4) for _, d in results],
    )
    return results


def vector_store_exists(store_dir: str) -> bool:
    store_path = Path(store_dir)
    return (store_path / _INDEX_FILE).exists() and (store_path / _META_FILE).exists()
