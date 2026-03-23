"""
app/core/models.py
------------------
Pydantic models that define the API contract and internal data structures.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Internal data structures ────────────────────────────────────────────────

class Chunk(BaseModel):
    """Represents one text chunk produced during ingestion."""
    chunk_id: str                    # e.g. "chunk_1"
    section_title: str               # e.g. "Attendance Policy"
    text: str                        # Full section text
    source_file: str                 # e.g. "campus_handbook.txt"


# ── API request / response ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Body of POST /chat"""
    question: str = Field(
        ...,
        min_length=1,
        description="The student's question (must not be empty).",
        examples=["What is the revaluation fee?"]
    )


class RetrievedSource(BaseModel):
    """One source chunk surfaced to the caller."""
    chunk_id: str
    section_title: str
    source_file: str


class ChatResponse(BaseModel):
    """Body returned by POST /chat"""
    answer: str
    sources: List[RetrievedSource]
    retrieved_chunks_count: int


# ── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
