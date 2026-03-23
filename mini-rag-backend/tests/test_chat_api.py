"""
tests/test_chat_api.py
-----------------------
Step 21  –  Basic tests for the Campus Help Assistant API.

Run with:
    pytest tests/ -v

Note: These tests mock the retrieval and LLM calls so they run
without a real API key or a pre-built vector store.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Patch the startup event so it doesn't try to load the vector store
with patch("app.services.retrieval_service.load_vector_store"):
    from app.main import app

client = TestClient(app)

# ── Fake data ─────────────────────────────────────────────────────────────────

FAKE_CHUNK_ATTENDANCE = {
    "chunk_id": "chunk_1",
    "section_title": "Attendance Policy",
    "text": (
        "Section 1: Attendance Policy\n"
        "Students must maintain at least 75 percent attendance in each subject "
        "to be eligible for semester examinations."
    ),
    "source_file": "campus_handbook.txt",
}

FAKE_RETRIEVED = [(FAKE_CHUNK_ATTENDANCE, 0.12)]  # low distance = highly relevant


# ── Test 1: Health endpoint ───────────────────────────────────────────────────

def test_health_endpoint():
    """GET /health must return 200 and {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── Test 2: Chat endpoint accepts valid request ───────────────────────────────

@patch("app.api.routes.retrieve_chunks", return_value=FAKE_RETRIEVED)
@patch("app.api.routes.call_llm", return_value="Students must maintain at least 75 percent attendance.")
def test_chat_valid_request(mock_llm, mock_retrieve):
    """POST /chat with a valid question must return 200 and a structured response."""
    response = client.post("/chat", json={"question": "What is the attendance rule?"})
    assert response.status_code == 200

    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert "retrieved_chunks_count" in body
    assert isinstance(body["sources"], list)
    assert body["retrieved_chunks_count"] == 1


# ── Test 3: Empty question is rejected ───────────────────────────────────────

def test_chat_empty_question():
    """POST /chat with an empty string must be rejected (422)."""
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


# ── Test 4: Known question returns grounded answer ───────────────────────────

@patch("app.api.routes.retrieve_chunks", return_value=FAKE_RETRIEVED)
@patch(
    "app.api.routes.call_llm",
    return_value="Students must maintain at least 75 percent attendance.",
)
def test_known_question_returns_grounded_answer(mock_llm, mock_retrieve):
    """A known question should get a non-empty, non-fallback answer."""
    response = client.post("/chat", json={"question": "What is the minimum attendance?"})
    body = response.json()

    assert response.status_code == 200
    assert "75 percent" in body["answer"]
    # Source should reference the Attendance Policy section
    assert any(s["section_title"] == "Attendance Policy" for s in body["sources"])


# ── Test 5: Unknown question returns no-answer response ───────────────────────

IRRELEVANT_CHUNK = {
    "chunk_id": "chunk_1",
    "section_title": "Attendance Policy",
    "text": "Section 1: Attendance Policy\nStudents must maintain at least 75 percent...",
    "source_file": "campus_handbook.txt",
}
# High distance = not relevant
IRRELEVANT_RETRIEVED = [(IRRELEVANT_CHUNK, 2.5)]


@patch("app.api.routes.retrieve_chunks", return_value=IRRELEVANT_RETRIEVED)
def test_unknown_question_returns_no_answer(mock_retrieve):
    """A question outside the knowledge base should return the no-answer message."""
    response = client.post("/chat", json={"question": "Who is the principal?"})
    body = response.json()

    assert response.status_code == 200
    assert "do not have enough information" in body["answer"].lower()
    # No sources should be attached because nothing was relevant
    assert body["sources"] == []
