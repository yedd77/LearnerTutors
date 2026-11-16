"""Split extracted document text into overlapping chunks for embedding."""
import re
from app.config import get_settings

# function to chunk text into smaller piece based on the chunk size and overlap
def chunk_sections(sections: list[dict]) -> list[dict]:
    """Return ``[{"page", "text"}]`` chunks of about CHUNK_SIZE characters per section.

    Whitespace runs are collapsed first. Consecutive chunks overlap by
    CHUNK_OVERLAP characters (capped at half the size; size is at least 200),
    and every chunk keeps its section's page number for citations.
    """
    s = get_settings()
    size = max(200, s.chunk_size)
    overlap = max(0, min(s.chunk_overlap, size // 2))
    step = size - overlap
    chunks = []
    for section in sections:
        text = re.sub(r"[ \t]+", " ", section["text"])
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        start = 0
        while start < len(text):
            part = text[start:start + size].strip()
            if part:
                chunks.append({"page": section.get("page"), "text": part})
            if start + size >= len(text):
                break
            start += step
    return chunks