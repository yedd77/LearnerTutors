"""Split extracted document text into overlapping, sentence-aligned chunks for embedding."""
import re
from app.config import get_settings

# A sentence ends at . ! or ? followed by whitespace and a likely sentence start.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _units(text: str) -> list[tuple[str, str]]:
    """Split text into ``(separator_before, sentence)`` units.

    Paragraph breaks become ``"\\n\\n"``, line breaks ``"\\n"`` (list items,
    DOCX paragraphs) and sentence breaks a space, so packing units back together
    reproduces the original layout.
    """
    units: list[tuple[str, str]] = []
    for p_index, paragraph in enumerate(re.split(r"\n\s*\n", text)):
        for l_index, line in enumerate(paragraph.split("\n")):
            for s_index, sentence in enumerate(_SENTENCE_END.split(line.strip())):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if not units:
                    separator = ""
                elif s_index:
                    separator = " "
                elif l_index:
                    separator = "\n"
                else:
                    separator = "\n\n" if p_index else " "
                units.append((separator, sentence))
    return units


def _split_long(sentence: str, size: int) -> list[str]:
    """Hard-split a unit longer than ``size`` at whitespace (or mid-word as a last resort)."""
    pieces = []
    while len(sentence) > size:
        cut = sentence.rfind(" ", 0, size + 1)
        if cut <= size // 2:
            cut = size
        pieces.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    if sentence:
        pieces.append(sentence)
    return pieces


def _join(units: list[tuple[str, str]]) -> str:
    return "".join((separator if i else "") + text for i, (separator, text) in enumerate(units))


def _pack(text: str, size: int, overlap: int) -> list[str]:
    """Pack sentences into chunks of at most ``size`` characters.

    Consecutive chunks repeat whole trailing sentences of up to ``overlap``
    characters, so a chunk never starts or ends mid-sentence unless a single
    sentence is longer than ``size``.
    """
    units: list[tuple[str, str]] = []
    for separator, sentence in _units(text):
        for i, piece in enumerate(_split_long(sentence, size)):
            units.append((separator if i == 0 else " ", piece))

    chunks: list[str] = []
    current: list[tuple[str, str]] = []
    for unit in units:
        if current and len(_join(current + [unit])) > size:
            chunks.append(_join(current))
            carried: list[tuple[str, str]] = []
            for previous in reversed(current):
                if len(_join([previous] + carried)) > overlap:
                    break
                carried.insert(0, previous)
            current = carried if len(_join(carried + [unit])) <= size else []
        current.append(unit)
    if current:
        chunks.append(_join(current))
    return chunks


# function to chunk text into smaller piece based on the chunk size and overlap
def chunk_sections(sections: list[dict]) -> list[dict]:
    """Return ``[{"page", "heading", "text", "chunk_index"}]`` chunks of up to CHUNK_SIZE characters.

    Whitespace runs are collapsed first. Text is split into paragraphs and
    sentences and packed into chunks; consecutive chunks of a section repeat
    whole sentences totalling at most CHUNK_OVERLAP characters (capped at half
    the size; size is at least 200). Every chunk keeps its section's page and
    heading (``None`` when absent) for citations, and ``chunk_index`` is its
    position in the whole document, used to fetch neighbours in order.
    """
    s = get_settings()
    size = max(200, s.chunk_size)
    overlap = max(0, min(s.chunk_overlap, size // 2))
    chunks = []
    for section in sections:
        text = re.sub(r"[ \t]+", " ", section["text"])
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        for part in _pack(text, size, overlap):
            part = part.strip()
            if part:
                chunks.append({
                    "page": section.get("page"),
                    "heading": section.get("heading"),
                    "text": part,
                    "chunk_index": len(chunks),
                })
    return chunks
