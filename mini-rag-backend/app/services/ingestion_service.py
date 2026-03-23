"""
app/services/ingestion_service.py
----------------------------------
Steps 5, 6, 7  –  Read raw text → chunk by section → save chunks.json
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List

from app.core.models import Chunk

logger = logging.getLogger(__name__)


# ── Step 5: Read raw file ────────────────────────────────────────────────────

def load_raw_text(file_path: str) -> str:
    """
    Open campus_handbook.txt and return its contents as a string.
    Raises FileNotFoundError or ValueError on bad input.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Knowledge-base file not found: {file_path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Knowledge-base file is empty: {file_path}")

    logger.info("Loaded raw text from %s (%d chars)", file_path, len(text))
    return text


# ── Step 6: Chunk by section ─────────────────────────────────────────────────

def split_into_chunks(raw_text: str, source_file: str = "campus_handbook.txt") -> List[Chunk]:
    """
    Split the handbook into one chunk per section.

    Each section starts with a line like:
        Section N: Title
    Everything between two such headers (or until end-of-file) is one chunk.
    """
    # Pattern matches lines like "Section 1: Attendance Policy"
    section_pattern = re.compile(
        r"(Section\s+\d+:\s+.+?)(?=Section\s+\d+:|END OF KNOWLEDGE BASE|$)",
        re.DOTALL | re.IGNORECASE,
    )

    raw_sections = section_pattern.findall(raw_text)

    if not raw_sections:
        raise ValueError("No sections found in knowledge-base text. Check the file format.")

    chunks: List[Chunk] = []
    for idx, section_text in enumerate(raw_sections, start=1):
        lines = section_text.strip().splitlines()

        # First line is the section header: "Section N: Title"
        header_line = lines[0].strip()
        # Extract just the title part after the colon
        title_match = re.match(r"Section\s+\d+:\s+(.+)", header_line, re.IGNORECASE)
        section_title = title_match.group(1).strip() if title_match else header_line

        # Body is everything after the first line
        body = "\n".join(lines[1:]).strip()
        full_text = f"{header_line}\n{body}"

        chunk = Chunk(
            chunk_id=f"chunk_{idx}",
            section_title=section_title,
            text=full_text,
            source_file=source_file,
        )
        chunks.append(chunk)
        logger.debug("  chunk_%d -> %s (%d chars)", idx, section_title, len(full_text))

    logger.info("Created %d chunks from handbook", len(chunks))
    return chunks


# ── Step 7: Persist chunks to JSON ──────────────────────────────────────────

def save_chunks(chunks: List[Chunk], output_path: str) -> None:
    """
    Serialize chunks to data/processed/chunks.json so the pipeline is
    traceable and can be inspected without re-running ingestion.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    data = [chunk.model_dump() for chunk in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d chunks → %s", len(chunks), output_path)


def load_chunks(chunks_path: str) -> List[Chunk]:
    """Load previously saved chunks.json back into Chunk objects."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**item) for item in data]
