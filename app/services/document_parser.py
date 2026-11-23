"""Extract plain text from uploaded course documents, keeping page/slide numbers and headings."""
from collections import Counter
from io import BytesIO
from pathlib import Path
import re
import fitz
from docx import Document
from pptx import Presentation
from app.config import get_settings

# File types the backend can index. The upload endpoints validate against this too.
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".txt", ".md"})

# PDF heading detection: a short line set noticeably larger than the body text.
_PDF_HEADING_SCALE = 1.2
_HEADING_MAX_WORDS = 15
_HEADING_MAX_CHARS = 200
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


class DocumentParseError(ValueError):
    """The file could not be read, is too long, or contains no extractable text."""


def _section(page: int | None, text: str, heading: str | None) -> dict:
    """One section; ``heading`` is included only when the document provides one."""
    section = {"page": page, "text": text}
    if heading:
        section["heading"] = heading[:_HEADING_MAX_CHARS]
    return section


def _looks_like_heading(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= _HEADING_MAX_WORDS and any(ch.isalpha() for ch in text)


def _pdf_sections(pdf, max_pages: int) -> list[dict]:
    """One section per page, split further where a heading starts.

    Headings are lines whose font is at least 1.2x the document's most common
    (body) size. The current heading carries over to the following pages.
    """
    pages: list[list[tuple[str, float]]] = []
    sizes: Counter = Counter()
    for n, page in enumerate(pdf, 1):
        if n > max_pages:
            raise DocumentParseError(f"PDF exceeds the {max_pages}-page limit.")
        lines: list[tuple[str, float]] = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:          # images and other non-text blocks
                continue
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
                if not spans:
                    continue
                text = "".join(span["text"] for span in line["spans"]).strip()
                size = max(float(span.get("size", 0)) for span in spans)
                lines.append((text, size))
                sizes[round(size, 1)] += len(text)
            if lines and lines[-1][0] != "":
                lines.append(("", 0.0))         # block boundary
        pages.append(lines)

    body_size = sizes.most_common(1)[0][0] if sizes else 0.0
    sections: list[dict] = []
    heading: str | None = None
    for n, lines in enumerate(pages, 1):
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            text = re.sub(r"\n{3,}", "\n\n", text)
            if text:
                sections.append(_section(n, text, heading))
            buffer.clear()

        for text, size in lines:
            if not text:
                buffer.append("")
                continue
            if body_size and size >= body_size * _PDF_HEADING_SCALE and _looks_like_heading(text):
                flush()
                heading = text
            buffer.append(text)
        flush()
    return sections


def _docx_sections(doc) -> list[dict]:
    """Paragraphs split at Heading/Title styles, then table rows as ``a | b | c``."""
    sections: list[dict] = []
    heading: str | None = None
    parts: list[str] = []

    def flush() -> None:
        if parts:
            sections.append(_section(None, "\n".join(parts), heading))
            parts.clear()

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name if paragraph.style is not None else "") or ""
        if style.startswith("Heading") or style == "Title":
            flush()
            heading = text
        parts.append(text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    flush()
    return sections


def _markdown_sections(text: str) -> list[dict]:
    """Split Markdown at ``#`` headings; each section keeps its heading line."""
    sections: list[dict] = []
    heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append(_section(None, body, heading))
        lines.clear()

    for line in text.splitlines():
        match = _MD_HEADING.match(line)
        if match:
            flush()
            heading = match.group(1)
        lines.append(line)
    flush()
    return sections


def extract_sections(filename: str, content: bytes) -> list[dict]:
    """Return the document's text as ``[{"page": int | None, "text": str, "heading"?: str}, ...]``.

    - PDF: one section per page with text (PyMuPDF), up to ``MAX_PAGES``, split
      where a larger-font heading starts.
    - PPTX: one section per slide, including text frames and table rows; the
      slide title (if any) is the heading.
    - DOCX: paragraphs split at Heading/Title styles, then table rows as ``a | b | c``.
    - MD: split at ``#`` headings. TXT: a single section. Both decoded as UTF-8
      (a BOM is ignored).

    ``heading`` is present only when the document provides one. Raises
    DocumentParseError for unsupported types, parser failures, the page limit,
    or when no text is found (scanned PDFs need OCR, which is not included).
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError("Supported formats: PDF, DOCX, PPTX, TXT, MD.")
    settings = get_settings()
    sections = []
    try:
        if ext == ".pdf":
            with fitz.open(stream=content, filetype="pdf") as pdf:
                sections = _pdf_sections(pdf, settings.max_pages)
        elif ext == ".docx":
            sections = _docx_sections(Document(BytesIO(content)))
        elif ext == ".pptx":
            pres = Presentation(BytesIO(content))
            for n, slide in enumerate(pres.slides, 1):
                parts = []
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False) and shape.text.strip():
                        parts.append(shape.text.strip())
                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            parts.append(" | ".join(c.text.strip() for c in row.cells))
                if parts:
                    title_shape = slide.shapes.title
                    title = title_shape.text.strip() if title_shape is not None else ""
                    sections.append(_section(n, "\n".join(parts), " ".join(title.split())))
        else:
            text = content.decode("utf-8-sig", errors="replace").strip()
            if text:
                sections = _markdown_sections(text) if ext == ".md" else [_section(None, text, None)]
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Could not parse '{filename}': {exc}") from exc
    if not sections:
        raise DocumentParseError("No extractable text found. Scanned PDF OCR is not included yet.")
    return sections
