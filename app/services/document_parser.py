"""Extract plain text from uploaded course documents, keeping page/slide numbers, headings and images."""
from collections import Counter
import hashlib
from io import BytesIO
import logging
from pathlib import Path
import re
import fitz
from docx import Document
from pptx import Presentation
from app.config import get_settings

logger = logging.getLogger(__name__)

# File types the backend can index. The upload endpoints validate against this too.
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".txt", ".md"})

# PDF heading detection: a short line set noticeably larger than the body text.
_PDF_HEADING_SCALE = 1.2
_HEADING_MAX_WORDS = 15
_HEADING_MAX_CHARS = 200
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
# Embedded images: a caption line near the image, and a cap on candidates per
# document (document_images filters size/repeats further before storing).
_CAPTION = re.compile(r"^(?:fig(?:ure)?\.?|table|diagram|chart|graph|illustration)\s*[\dIVX]*[.:)\-\s]",
                      re.IGNORECASE)
_CAPTION_MAX_CHARS = 200
# Office tools fill missing alt text with the file name ("image.png"): not a caption.
_FILE_NAME = re.compile(r"^[\w\-. ]+\.(?:png|jpe?g|gif|bmp|tiff?|emf|wmf|svg|webp)$", re.IGNORECASE)
MAX_IMAGE_CANDIDATES = 400

# Vector-drawn PDF figures (charts and diagrams made of paths, not pictures) are
# rendered to PNG when a figure caption has no picture. Tables are excluded:
# their content is already extracted as text.
_FIGURE_CAPTION = re.compile(r"^(?:fig(?:ure)?\.?|diagram|chart|graph|illustration)\s*[\dIVX]*(?:[.:)\-\s]|$)",
                             re.IGNORECASE)
_FIGURE_SEARCH_GAP = 60.0      # pt between a caption and the first drawing of its figure
_FIGURE_JOIN_GAP = 15.0        # pt between drawings of the same figure
_FIGURE_LABEL_MARGIN = 20.0    # pt around the drawings where labels (axis titles, legends) count
_FIGURE_MARGIN = 4.0           # pt added around the rendered region
_FIGURE_MIN_W, _FIGURE_MIN_H = 100.0, 60.0
_FIGURE_MAX_PAGE_SHARE = 0.9
_FIGURE_MIN_PATH_ITEMS = 3
_FIGURE_ZOOM = 2.0
_FIGURE_MAX_PIXELS_SIDE = 1600   # same as document_images.MAX_SIDE
_BODY_LINE_MIN_WORDS = 8
_MAX_FIGURES_PER_PAGE = 10


class DocumentParseError(ValueError):
    """The file could not be read, is too long, or contains no extractable text."""


class _Images:
    """Collects embedded image occurrences for one document, in reading order."""

    def __init__(self) -> None:
        self.count = 0

    def make(self, data: bytes, offset: int, caption: str | None, page: int | None) -> dict | None:
        """One occurrence; ``offset`` is its character position in the section text."""
        if not data or self.count >= MAX_IMAGE_CANDIDATES:
            return None
        self.count += 1
        caption = " ".join((caption or "").split())[:_CAPTION_MAX_CHARS] or None
        return {"ref": self.count, "hash": hashlib.sha256(data).hexdigest(), "data": data,
                "offset": offset, "caption": caption, "page": page}


def _section(page: int | None, text: str, heading: str | None, images: list[dict] | None = None) -> dict:
    """One section; ``heading`` and ``images`` are included only when present."""
    section = {"page": page, "text": text}
    if heading:
        section["heading"] = heading[:_HEADING_MAX_CHARS]
    if images:
        section["images"] = images
    return section


def _alt_text(values: list[str]) -> str | None:
    """The first real alt text (not blank, not just a file name)."""
    for value in values:
        value = value.strip()
        if value and not _FILE_NAME.match(value):
            return value
    return None


def _looks_like_heading(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= _HEADING_MAX_WORDS and any(ch.isalpha() for ch in text)


def _caption_index_near(items: list[tuple], position: int) -> int | None:
    """Index of the "Figure 3: ..."-style line just after (preferred) or before an image."""
    after = [i for i in range(position + 1, min(len(items), position + 4))
             if items[i][0] == "text" and items[i][1]]
    before = [i for i in range(position - 1, max(0, position - 2) - 1, -1)
              if items[i][0] == "text" and items[i][1]]
    for i in after + before:
        if _CAPTION.match(items[i][1]):
            return i
    return None


_CAPTION_CONTINUATION_LINES = 3


def _caption_near(items: list[tuple], position: int) -> str | None:
    """A "Figure 3: ..."-style caption just after (preferred) or before an image.

    Books often put only the label on the first line ("Figure 7.5") and the
    description on the next ones ("POP blind rivet: standard open type"), so
    up to 3 following lines of the same text block are included. Without them
    the model only sees "Figure 7.5" and cannot tell what the figure shows.
    """
    index = _caption_index_near(items, position)
    if index is None:
        return None
    parts = [items[index][1]]
    for item in items[index + 1:index + 1 + _CAPTION_CONTINUATION_LINES]:
        if item[0] != "text" or not item[1] or _CAPTION.match(item[1]):
            break                       # block boundary, an image, or the next caption
        if len(" ".join(parts + [item[1]])) > _CAPTION_MAX_CHARS:
            break
        parts.append(item[1])
    return " ".join(parts)


def _path_bounds(drawing: dict):
    """Bounding box of a drawing's path items.

    Computed from the points, because the ``rect`` PyMuPDF reports for a path
    made of several separate lines can miss some of them (e.g. a chart axis).
    """
    points = []
    for item in drawing.get("items") or []:
        for value in item[1:]:
            if isinstance(value, fitz.Point):
                points.append(value)
            elif isinstance(value, fitz.Rect):
                points += [value.tl, value.br]
            elif isinstance(value, fitz.Quad):
                points += [value.ul, value.ur, value.ll, value.lr]
    if not points:
        rect = drawing.get("rect")
        return fitz.Rect(rect) if rect is not None else None
    return fitz.Rect(min(p.x for p in points), min(p.y for p in points),
                     max(p.x for p in points), max(p.y for p in points))


def _figure_drawings(page) -> list[tuple]:
    """``(rect, path item count)`` of the page's drawings, without page furniture.

    Furniture: thin rules spanning most of the page width (header/footer lines)
    and borders covering most of the page.
    """
    width, area = page.rect.width, page.rect.width * page.rect.height
    drawings = []
    for drawing in page.get_drawings():
        rect = _path_bounds(drawing)
        if rect is None:
            continue
        if rect.is_empty and rect.width == 0 and rect.height == 0:
            continue
        if min(rect.width, rect.height) < 2 and rect.width > 0.8 * width:
            continue
        if rect.width * rect.height > _FIGURE_MAX_PAGE_SHARE * area:
            continue
        drawings.append((rect, len(drawing.get("items") or [])))
    return drawings


def _union(a, b):
    """Bounding box of two rects. Unlike ``Rect | Rect`` it keeps zero-height or
    zero-width rects (a horizontal axis line is "empty" to PyMuPDF)."""
    return fitz.Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def _gap(a, b) -> float:
    """Distance between two rects (0 when they touch or overlap)."""
    dx = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return max(dx, dy)


def _inside_share(inner, outer) -> float:
    """Share of ``inner``'s area that lies inside ``outer``."""
    area = inner.width * inner.height
    if area <= 0:
        return 1.0 if outer.contains(inner.tl) and outer.contains(inner.br) else 0.0
    overlap = fitz.Rect(inner) & outer
    return 0.0 if overlap.is_empty else overlap.width * overlap.height / area


def _figure_region(page, caption_rect, drawings, body_lines) -> tuple | None:
    """The union of the drawings forming the figure next to a caption, or None.

    Looks above the caption first (where this kind of book puts figures), then
    below. Starts from the drawing nearest the caption in the same column and
    adds drawings within a small gap, never growing over a body-text line.
    The column is the caption's width widened by the body lines that overlap it
    horizontally: the full text width on a one-column page, one column otherwise
    (a caption like "FIGURE 2.4" alone is much narrower than its figure).
    """
    column = fitz.Rect(caption_rect.x0 - 20, 0, caption_rect.x1 + 20, page.rect.height)
    for line in body_lines:
        if line.x0 < caption_rect.x1 and line.x1 > caption_rect.x0:
            column = fitz.Rect(min(column.x0, line.x0 - 20), 0, max(column.x1, line.x1 + 20),
                               page.rect.height)

    def in_column(rect) -> bool:
        centre = (rect.x0 + rect.x1) / 2
        overlap = min(rect.x1, column.x1) - max(rect.x0, column.x0)
        return column.x0 <= centre <= column.x1 or overlap >= 0.5 * max(rect.width, 1)

    for above in (True, False):
        side = [(rect, count) for rect, count in drawings if in_column(rect) and (
            rect.y1 <= caption_rect.y0 + 2 if above else rect.y0 >= caption_rect.y1 - 2)]
        if not side:
            continue
        def distance(entry):
            rect = entry[0]
            return caption_rect.y0 - rect.y1 if above else rect.y0 - caption_rect.y1
        seed = min(side, key=distance)
        if distance(seed) > _FIGURE_SEARCH_GAP:
            continue
        region, items, members = fitz.Rect(seed[0]), seed[1], [seed]
        grown = True
        while grown:
            grown = False
            for entry in side:
                if any(entry is m for m in members) or _gap(entry[0], region) > _FIGURE_JOIN_GAP:
                    continue
                candidate = _union(region, entry[0])
                if any(_inside_share(line, candidate) > 0.5 for line in body_lines):
                    continue
                region, items, grown = candidate, items + entry[1], True
                members.append(entry)
        if (items >= _FIGURE_MIN_PATH_ITEMS and region.width >= _FIGURE_MIN_W
                and region.height >= _FIGURE_MIN_H):
            return region, above
    return None


def _render_figure(page, region) -> bytes:
    """PNG of a page region at 2x, scaled down so the longest side stays within 1600 px."""
    longest = max(region.width, region.height, 1.0)
    zoom = min(_FIGURE_ZOOM, _FIGURE_MAX_PIXELS_SIDE / longest)
    pixmap = page.get_pixmap(clip=region, matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pixmap.tobytes("png")


def _vector_figures(page, items: list[tuple], page_number: int, images: "_Images") -> list[tuple]:
    """Add rendered vector figures to a page's reading-order ``items``.

    A figure caption ("Figure 2 ...", not "Table") that no picture already uses
    gets its nearby drawings rendered as an image, inserted just before the
    caption line so it is anchored like an embedded picture. A picture lying
    inside such a region (a raster with vector annotations) is replaced by the
    rendered region. Label text stays in the page text. Failures are logged
    per figure and never stop parsing. Rotated pages are skipped.
    """
    captions = [i for i, item in enumerate(items)
                if item[0] == "text" and item[1] and _FIGURE_CAPTION.match(item[1])]
    if not captions or images.count >= MAX_IMAGE_CANDIDATES:
        return items
    if page.rotation:
        # Text and drawing coordinates don't line up on rotated (landscape) pages;
        # rendering a guessed region could show the wrong part of the page.
        return items
    try:
        drawings = _figure_drawings(page)
    except Exception:
        logger.warning("document.vector_figures.drawings_failed page=%s", page_number, exc_info=True)
        return items
    if not drawings:
        return items

    rasters = {i: _caption_index_near(items, i) for i, item in enumerate(items) if item[0] == "image"}
    text_lines = [(i, item) for i, item in enumerate(items) if item[0] == "text" and item[1]]
    page_area = page.rect.width * page.rect.height
    inserts: dict[int, dict] = {}
    dropped: set[int] = set()
    for index in captions[:_MAX_FIGURES_PER_PAGE]:
        caption_rect = fitz.Rect(items[index][3])
        body_lines = [fitz.Rect(item[3]) for i, item in text_lines if i != index
                      and len(item[1].split()) >= _BODY_LINE_MIN_WORDS
                      and fitz.Rect(item[3]).width >= 0.5 * caption_rect.width]
        found = _figure_region(page, caption_rect, drawings, body_lines)
        if found is None:
            continue
        region, _ = found
        # Labels and legends next to the drawings belong to the figure.
        reach = fitz.Rect(region.x0 - _FIGURE_LABEL_MARGIN, region.y0 - _FIGURE_LABEL_MARGIN,
                          region.x1 + _FIGURE_LABEL_MARGIN, region.y1 + _FIGURE_LABEL_MARGIN)
        for i, item in text_lines:
            line = fitz.Rect(item[3])
            if (i != index and not _FIGURE_CAPTION.match(item[1])
                    and len(item[1].split()) < _BODY_LINE_MIN_WORDS and _inside_share(line, reach) > 0.6):
                region = _union(region, line)
        region = (region + (-_FIGURE_MARGIN, -_FIGURE_MARGIN, _FIGURE_MARGIN, _FIGURE_MARGIN)) & page.rect
        if region.width * region.height > _FIGURE_MAX_PAGE_SHARE * page_area:
            continue
        inside = [i for i in rasters if i not in dropped and items[i][1].get("bbox")
                  and _inside_share(fitz.Rect(items[i][1]["bbox"]), region) > 0.8]
        if any(caption == index for i, caption in rasters.items() if i not in inside):
            continue                       # a picture outside the drawings already has this caption
        if images.count + len(inserts) >= MAX_IMAGE_CANDIDATES:
            break
        try:
            png = _render_figure(page, region)
        except Exception:
            logger.warning("document.vector_figures.render_failed page=%s", page_number, exc_info=True)
            continue
        dropped.update(inside)
        inserts[index] = {"image": png, "bbox": tuple(region), "rendered": True}

    if not inserts:
        return items
    result: list[tuple] = []
    for i, item in enumerate(items):
        if i in dropped:
            continue
        if i in inserts:
            result.append(("image", inserts[i]))
        result.append(item)
    return result


def _pdf_sections(pdf, max_pages: int, images: _Images) -> list[dict]:
    """One section per page, split further where a heading starts, with image positions.

    Headings are lines whose font is at least 1.2x the document's most common
    (body) size. The current heading carries over to the following pages.
    """
    pages: list[list[tuple]] = []
    sizes: Counter = Counter()
    for n, page in enumerate(pdf, 1):
        if n > max_pages:
            raise DocumentParseError(f"PDF exceeds the {max_pages}-page limit.")
        # ("text", line, size, bbox) | ("text", "", 0.0, None) block boundary | ("image", block)
        items: list[tuple] = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") == 1:
                items.append(("image", block))
                continue
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
                if not spans:
                    continue
                text = "".join(span["text"] for span in line["spans"]).strip()
                size = max(float(span.get("size", 0)) for span in spans)
                items.append(("text", text, size, tuple(line.get("bbox") or (0, 0, 0, 0))))
                sizes[round(size, 1)] += len(text)
            if items and items[-1][0] == "text" and items[-1][1]:
                items.append(("text", "", 0.0, None))      # block boundary
        pages.append(_vector_figures(page, items, n, images))

    body_size = sizes.most_common(1)[0][0] if sizes else 0.0
    sections: list[dict] = []
    heading: str | None = None
    for n, items in enumerate(pages, 1):
        lines: list[str] = []
        placed: list[tuple[int, dict]] = []      # (line position, image block)

        def flush() -> None:
            raw = "\n".join(lines)
            lead = len(raw) - len(raw.lstrip())
            text = raw.strip()
            found = []
            for line_count, (block, caption) in placed:
                offset = len("\n".join(lines[:line_count])) - lead
                occurrence = images.make(block.get("image") or b"", min(max(offset, 0), len(text)),
                                         caption, n)
                if occurrence:
                    found.append(occurrence)
            if text or found:
                sections.append(_section(n, text, heading, found))
            lines.clear()
            placed.clear()

        for position, item in enumerate(items):
            if item[0] == "image":
                placed.append((len(lines), (item[1], _caption_near(items, position))))
                continue
            _, text, size, _bbox = item
            if not text:
                lines.append("")
                continue
            if body_size and size >= body_size * _PDF_HEADING_SCALE and _looks_like_heading(text):
                flush()
                heading = text
            lines.append(text)
        flush()
    # A page with images but no text still anchors its images; drop text-less
    # sections only after images were collected.
    return [s for s in sections if s["text"] or s.get("images")]


def _docx_sections(doc, images: _Images) -> list[dict]:
    """Paragraphs split at Heading/Title styles, then table rows as ``a | b | c``; inline images kept."""
    sections: list[dict] = []
    heading: str | None = None
    parts: list[str] = []
    found: list[dict] = []

    def flush() -> None:
        if parts or found:
            sections.append(_section(None, "\n".join(parts), heading, list(found)))
            parts.clear()
            found.clear()

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        style = (paragraph.style.name if paragraph.style is not None else "") or ""
        if text and (style.startswith("Heading") or style == "Title"):
            flush()
            heading = text
        if text:
            parts.append(text)
        element = paragraph._p
        alt = _alt_text(element.xpath(".//wp:docPr/@descr"))
        for r_id in element.xpath(".//a:blip/@r:embed"):
            part = doc.part.related_parts.get(r_id)
            caption = text if text and _CAPTION.match(text) else alt
            occurrence = images.make(getattr(part, "blob", b""), len("\n".join(parts)), caption, None)
            if occurrence:
                found.append(occurrence)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    flush()
    return sections


def _pptx_sections(pres, images: _Images) -> list[dict]:
    """One section per slide (text frames and tables); the slide title is the heading."""
    sections: list[dict] = []
    for n, slide in enumerate(pres.slides, 1):
        title_shape = slide.shapes.title
        title = " ".join(title_shape.text.split()) if title_shape is not None else ""
        parts: list[str] = []
        found: list[dict] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text.strip():
                parts.append(shape.text.strip())
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
            try:
                blob = shape.image.blob          # pictures and picture placeholders
            except (AttributeError, ValueError, KeyError):
                continue
            caption = _alt_text(shape._element.xpath(".//p:cNvPr/@descr")) or title
            occurrence = images.make(blob, len("\n".join(parts)), caption, n)
            if occurrence:
                found.append(occurrence)
        if parts or found:
            sections.append(_section(n, "\n".join(parts), title, found))
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
    """Return the document's text as ``[{"page", "text", "heading"?, "images"?}, ...]``.

    - PDF: one section per page with text (PyMuPDF), up to ``MAX_PAGES``, split
      where a larger-font heading starts.
    - PPTX: one section per slide, including text frames and table rows; the
      slide title (if any) is the heading.
    - DOCX: paragraphs split at Heading/Title styles, then table rows as ``a | b | c``.
    - MD: split at ``#`` headings. TXT: a single section. Both decoded as UTF-8
      (a BOM is ignored).

    ``heading`` is present only when the document provides one. ``images`` lists
    embedded images (PDF, DOCX, PPTX) as ``{"ref", "hash", "data", "offset",
    "caption", "page"}``, where ``offset`` is the image's character position in
    ``text``; see document_images for filtering and storage. Raises
    DocumentParseError for unsupported types, parser failures, the page limit,
    or when no text is found (scanned PDFs need OCR, which is not included).
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError("Supported formats: PDF, DOCX, PPTX, TXT, MD.")
    settings = get_settings()
    images = _Images()
    sections = []
    try:
        if ext == ".pdf":
            with fitz.open(stream=content, filetype="pdf") as pdf:
                sections = _pdf_sections(pdf, settings.max_pages, images)
        elif ext == ".docx":
            sections = _docx_sections(Document(BytesIO(content)), images)
        elif ext == ".pptx":
            sections = _pptx_sections(Presentation(BytesIO(content)), images)
        else:
            text = content.decode("utf-8-sig", errors="replace").strip()
            if text:
                sections = _markdown_sections(text) if ext == ".md" else [_section(None, text, None)]
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Could not parse '{filename}': {exc}") from exc
    if not any(section["text"] for section in sections):
        raise DocumentParseError("No extractable text found. Scanned PDF OCR is not included yet.")
    return sections
