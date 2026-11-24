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


def _normalise(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _section_units(section: dict, size: int) -> list[tuple]:
    """Text units ``(separator, sentence)`` with image units ``(None, ref)`` at their positions."""
    text = section["text"]
    images = sorted(section.get("images") or [], key=lambda image: image["offset"])
    units: list[tuple] = []
    start = 0
    for image in images + [None]:
        end = len(text) if image is None else min(max(image["offset"], start), len(text))
        for i, (separator, sentence) in enumerate(_units(_normalise(text[start:end]))):
            if i == 0 and any(u[0] is not None for u in units):
                separator = "\n"            # text continues after an image
            for j, piece in enumerate(_split_long(sentence, size)):
                units.append((separator if j == 0 else " ", piece))
        if image is not None:
            units.append((None, image["ref"]))
        start = end
    return units


def _pack(units: list[tuple], size: int, overlap: int) -> tuple[list[str], list[tuple[int, int]]]:
    """Pack sentences into chunks of at most ``size`` characters; place images between them.

    Consecutive chunks repeat whole trailing sentences of up to ``overlap``
    characters, so a chunk never starts or ends mid-sentence unless a single
    sentence is longer than ``size``. Returns the chunk texts and ``(ref, anchor)``
    pairs: the image follows chunk ``anchor`` (``-1``: before the first chunk).
    An image inside a chunk's new text is anchored to that chunk.
    """
    chunks: list[str] = []
    anchors: list[tuple[int, int]] = []
    current: list[tuple[str, str]] = []
    fresh = 0                                   # units added since the last chunk was emitted
    for unit in units:
        if unit[0] is None:
            anchors.append((unit[1], len(chunks) if fresh else len(chunks) - 1))
            continue
        if current and len(_join(current + [unit])) > size:
            chunks.append(_join(current))
            carried: list[tuple[str, str]] = []
            for previous in reversed(current):
                if len(_join([previous] + carried)) > overlap:
                    break
                carried.insert(0, previous)
            current = carried if len(_join(carried + [unit])) <= size else []
            fresh = 0
        current.append(unit)
        fresh += 1
    if current and fresh:
        chunks.append(_join(current))
    return chunks, anchors


# function to chunk text into smaller piece based on the chunk size and overlap
def chunk_sections(sections: list[dict]) -> list[dict]:
    """Return ``[{"page", "heading", "text", "chunk_index"}]`` chunks of up to CHUNK_SIZE characters.

    Whitespace runs are collapsed first. Text is split into paragraphs and
    sentences and packed into chunks; consecutive chunks of a section repeat
    whole sentences totalling at most CHUNK_OVERLAP characters (capped at half
    the size; size is at least 200). Every chunk keeps its section's page and
    heading (``None`` when absent) for citations, and ``chunk_index`` is its
    position in the whole document, used to fetch neighbours in order.

    Images in a section (see document_parser) are anchored without entering
    the text: ``image_refs`` on a chunk lists images that follow it, and
    ``image_refs_before`` on the first chunk lists images before any text. Both
    keys are present only when there are images.
    """
    s = get_settings()
    size = max(200, s.chunk_size)
    overlap = max(0, min(s.chunk_overlap, size // 2))
    chunks: list[dict] = []
    before: list[int] = []                      # images before the document's first chunk
    for section in sections:
        base = len(chunks)
        texts, anchors = _pack(_section_units(section, size), size, overlap)
        for part in texts:
            chunks.append({
                "page": section.get("page"),
                "heading": section.get("heading"),
                "text": part.strip(),
                "chunk_index": len(chunks),
            })
        for ref, anchor in anchors:
            index = base + anchor
            if index < 0:
                before.append(ref)
            else:
                chunks[index].setdefault("image_refs", []).append(ref)
    if before and chunks:
        chunks[0]["image_refs_before"] = before
    return chunks
