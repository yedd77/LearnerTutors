import re
from app.config import get_settings

# function to chunk text into smaller piece based on the chunk size and overlap
def chunk_sections(sections: list[dict]) -> list[dict]:
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