"""
app/services/prompt_service.py
--------------------------------
Step 15  –  Build a grounded prompt from retrieved chunks.
"""

from typing import List, Tuple


# ── Step 14: Fallback response ────────────────────────────────────────────────

NO_ANSWER_RESPONSE = (
    "I do not have enough information in the provided knowledge base to answer that."
)

# L2 distance threshold — anything above this is considered "not relevant enough".
# (all-MiniLM-L6-v2 distances typically range 0 – 4 for unrelated pairs)
RELEVANCE_THRESHOLD = 1.5


def is_relevant(distance: float) -> bool:
    """Return True if the distance is close enough to be considered relevant."""
    return distance < RELEVANCE_THRESHOLD


# ── Step 15: Build prompt ────────────────────────────────────────────────────

def build_prompt(
    question: str,
    retrieved: List[Tuple[dict, float]],
) -> Tuple[str, str]:
    """
    Build a (system_prompt, user_message) pair for the LLM.

    Returns (system_prompt, user_message).
    If no relevant chunks are found, user_message will be empty string and
    the caller should return NO_ANSWER_RESPONSE directly.
    """
    # Filter by relevance
    relevant = [(meta, dist) for meta, dist in retrieved if is_relevant(dist)]

    if not relevant:
        return "", ""

    # Build context block
    context_parts = []
    for meta, _ in relevant:
        context_parts.append(
            f"[{meta['section_title']}]\n{meta['text']}"
        )
    context_block = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful Campus Help Assistant.\n"
        "Answer the student's question ONLY using the provided context below.\n"
        "If the answer cannot be found in the context, reply exactly with:\n"
        f'"{NO_ANSWER_RESPONSE}"\n'
        "Keep your answer concise, factual, and mention the source section name."
    )

    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    return system_prompt, user_message
