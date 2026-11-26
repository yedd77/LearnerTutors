"""Render a ``StudySheet`` as an A4 revision sheet PDF.

Layout (one page): title with a teal rule, the overview as a lead paragraph,
the other sections as open ruled blocks in two balanced columns ("Remember
this" as the one coral callout), a full-width tinted summary, and a footer with
sources, page number and branding. Only sections with content appear.

Fitting: the page is tried at a few decreasing scales, but body text never
goes below ``Theme.min_body_size``. If it still does not fit, nothing is cut:
the sheet is laid out in one column across as many pages as needed, and the
page count is returned so the caller can tell the student.
"""
import io
import logging
from dataclasses import dataclass
from datetime import datetime

from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import BaseDocTemplate, Frame, NextPageTemplate, PageTemplate, Paragraph, Spacer

from app.pdf import components as ui
from app.pdf.theme import DEFAULT_THEME, Theme
from app.schemas import SourceItem
from app.services.study_sheet import StudySheet

logger = logging.getLogger(__name__)

# Largest first: short sheets grow to fill the page, long ones shrink (to a floor).
SCALES = (1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8)
COLUMN_SECTIONS = ("key_concepts", "definitions", "main_points", "examples", "formulas", "remember")
MAX_SOURCE_FILES = 8


@dataclass(frozen=True)
class RenderResult:
    """The PDF and how many pages it has."""
    pdf: bytes
    pages: int
    scale: float


# --------------------------------------------------------------------------- content
def _rows(key: str, sheet: StudySheet, theme: Theme, styles: ui.Styles) -> list:
    """Table rows (Paragraphs) for one section; empty when the section has no content."""
    section = theme.sections[key]
    bullet = ui.accent_bullet_style(section, theme, styles)
    muted = theme.muted

    def t(value: str) -> str:
        return ui.rich(value, styles.fonts)

    if key == "overview":
        return [Paragraph(t(sheet.overview), styles.lead)] if sheet.overview else []
    # Definitions and formulas lead with a bold term, so they need no bullet.
    if key == "definitions":
        return [Paragraph(f"<b>{t(d.term)}:</b> {t(d.meaning)}", styles.body)
                for d in sheet.definitions]
    if key == "formulas":
        # Extra leading so sub/superscripts do not touch the meaning line below.
        formula = ParagraphStyle("formula", parent=styles.body,
                                 leading=styles.body.fontSize * 1.55)
        return [Paragraph(f"<b>{t(f.expression)}</b>"
                          + (f"<br/><font color='{muted}'>{t(f.meaning)}</font>"
                             if f.meaning else ""), formula)
                for f in sheet.formulas]
    items = getattr(sheet, key)
    return [Paragraph(t(item), bullet, bulletText="•") for item in items]


def _cards(sheet: StudySheet, theme: Theme, styles: ui.Styles, keys, width: float) -> list:
    """Cards for the given section keys that have content.

    The overview is a bare lead paragraph (no heading): it is read first and
    needs no label.
    """
    cards = []
    for key in keys:
        rows = _rows(key, sheet, theme, styles)
        if not rows:
            continue
        cards.append(rows[0] if key == "overview"
                     else ui.section_card(theme.sections[key], rows, width, theme, styles))
    return cards


def sources_text(sources: list[SourceItem]) -> str:
    """Footer text listing source files and pages, e.g. ``Sources: a.pdf (p. 2, 5)``."""
    pages: dict[str, set[int]] = {}
    for source in sources:
        entry = pages.setdefault(source.filename or "Unknown document", set())
        if source.page:
            entry.add(source.page)
    if not pages:
        return "<b>Sources:</b> course materials"
    parts = []
    for name, numbers in list(pages.items())[:MAX_SOURCE_FILES]:
        label = ui.text(name)
        if numbers:
            label += " (p. " + ", ".join(str(n) for n in sorted(numbers)) + ")"
        parts.append(label)
    more = len(pages) - MAX_SOURCE_FILES
    if more > 0:
        parts.append(f"and {more} more document" + ("s" if more != 1 else ""))
    return "<b>Sources:</b> " + " · ".join(parts)


def _subtitle(generated_at: datetime) -> str:
    """Header subtitle with the generation date."""
    return f"Study sheet · Generated {generated_at.day} {generated_at:%B %Y}"


# --------------------------------------------------------------------------- one page
def _stack_height(heights: list[float], gap: float) -> float:
    """Height of cards stacked with ``gap`` between them."""
    return sum(heights) + gap * max(0, len(heights) - 1)


def _split_columns(heights: list[float], gap: float) -> int:
    """Index splitting cards (in order) into two columns of the most similar height."""
    best, best_height = len(heights), float("inf")
    for i in range(1, len(heights)):
        tallest = max(_stack_height(heights[:i], gap), _stack_height(heights[i:], gap))
        if tallest < best_height:
            best, best_height = i, tallest
    return best


def _try_one_page(sheet: StudySheet, sources: list[SourceItem], generated_at: datetime,
                  theme: Theme, scale: float) -> bytes | None:
    """The one-page PDF at ``scale``, or None if the content does not fit."""
    styles = ui.make_styles(theme, scale)
    page_w, page_h = theme.page_size
    width = page_w - 2 * theme.margin
    gap = theme.gutter * (0.6 + 0.4 * scale)
    col_w = (width - theme.gutter) / 2

    buffer = io.BytesIO()
    c = pdf_canvas.Canvas(buffer, pagesize=theme.page_size)
    _set_metadata(c, sheet, theme)

    footer = ui.footer_paragraph(sources_text(sources), theme, styles)
    footer_h = ui.footer_height(footer, theme, styles, width)
    top = page_h - theme.margin
    header_h = ui.draw_header(c, theme, styles, sheet.title, _subtitle(generated_at), top, width)
    body_top = top - header_h - gap
    body_bottom = theme.margin + footer_h + gap * 0.6
    available = body_top - body_bottom

    overview = _cards(sheet, theme, styles, ("overview",), width)
    summary = _cards(sheet, theme, styles, ("summary",), width)
    column_cards = _cards(sheet, theme, styles, COLUMN_SECTIONS, col_w)
    if len(column_cards) == 1:  # a lone card reads better full width
        column_cards = _cards(sheet, theme, styles, COLUMN_SECTIONS, width)

    def measure(card, w):
        return card.wrapOn(c, w, available)[1]

    overview_h = [measure(card, width) for card in overview]
    summary_h = [measure(card, width) for card in summary]
    if len(column_cards) == 1:
        left, right, split = column_cards, [], 1
        left_h, right_h = [measure(column_cards[0], width)], []
    else:
        heights = [measure(card, col_w) for card in column_cards]
        split = _split_columns(heights, gap) if column_cards else 0
        left, right = column_cards[:split], column_cards[split:]
        left_h, right_h = heights[:split], heights[split:]

    blocks = [h for h in (_stack_height(overview_h, gap),
                          max(_stack_height(left_h, gap), _stack_height(right_h, gap)),
                          _stack_height(summary_h, gap)) if h > 0]
    if _stack_height(blocks, gap) > available + 0.5:
        return None

    y = body_top
    for card, h in zip(overview, overview_h):
        card.drawOn(c, theme.margin, y - h)
        y -= h + gap
    column_x = (theme.margin, theme.margin + col_w + theme.gutter)
    lowest = y
    for x, cards, heights in ((column_x[0], left, left_h), (column_x[1], right, right_h)):
        cy = y
        for card, h in zip(cards, heights):
            card.drawOn(c, x, cy - h)
            cy -= h + gap
        lowest = min(lowest, cy)
    y = lowest
    for card, h in zip(summary, summary_h):
        card.drawOn(c, theme.margin, y - h)
        y -= h + gap

    ui.draw_footer(c, theme, styles, footer, 1, 1, width)
    c.showPage()
    c.save()
    return buffer.getvalue()


# --------------------------------------------------------------------------- several pages
class _CountingDoc(BaseDocTemplate):
    """Doc template that records how many pages were produced."""
    pages = 0

    def afterPage(self):
        """Count each finished page."""
        self.pages = self.page


def _multi_page(sheet: StudySheet, sources: list[SourceItem], generated_at: datetime,
                theme: Theme) -> tuple[bytes, int]:
    """One-column layout across as many pages as the content needs, at full size."""
    styles = ui.make_styles(theme, 1.0)
    page_w, page_h = theme.page_size
    width = page_w - 2 * theme.margin
    footer = ui.footer_paragraph(sources_text(sources), theme, styles)
    probe = pdf_canvas.Canvas(io.BytesIO(), pagesize=theme.page_size)
    footer_h = ui.footer_height(footer, theme, styles, width)
    header_h = ui.draw_header(probe, theme, styles, sheet.title, _subtitle(generated_at),
                              page_h - theme.margin, width)
    bottom = theme.margin + footer_h + theme.gutter * 0.6

    def build(total: int) -> tuple[bytes, int]:
        buffer = io.BytesIO()
        doc = _CountingDoc(buffer, pagesize=theme.page_size, leftMargin=theme.margin,
                           rightMargin=theme.margin, topMargin=theme.margin,
                           bottomMargin=bottom, title=sheet.title, author=theme.brand_name)

        def first(c, d):
            _set_metadata(c, sheet, theme)
            ui.draw_header(c, theme, styles, sheet.title, _subtitle(generated_at),
                           page_h - theme.margin, width)
            ui.draw_footer(c, theme, styles, footer, d.page, total, width)

        def later(c, d):
            ui.draw_footer(c, theme, styles, footer, d.page, total, width)

        first_frame = Frame(theme.margin, bottom, width,
                            page_h - theme.margin - header_h - theme.gutter - bottom,
                            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        later_frame = Frame(theme.margin, bottom, width, page_h - theme.margin - bottom,
                            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        doc.addPageTemplates([PageTemplate("first", [first_frame], onPage=first),
                              PageTemplate("later", [later_frame], onPage=later)])
        story = [NextPageTemplate("later")]
        keys = ("overview",) + COLUMN_SECTIONS + ("summary",)
        for card in _cards(sheet, theme, styles, keys, width):
            story += [card, Spacer(1, theme.gutter * 0.8)]
        doc.build(story)
        return buffer.getvalue(), doc.pages

    _, pages = build(0)
    return build(pages)


# --------------------------------------------------------------------------- entry point
def _set_metadata(c, sheet: StudySheet, theme: Theme) -> None:
    """PDF document properties."""
    c.setTitle(sheet.title)
    c.setAuthor(theme.brand_name)
    c.setSubject("Study sheet")
    c.setCreator(theme.brand_name)


def render_study_sheet(sheet: StudySheet, sources: list[SourceItem],
                       generated_at: datetime | None = None,
                       theme: Theme = DEFAULT_THEME) -> RenderResult:
    """Render ``sheet`` to PDF bytes, on one page when it fits at a readable size."""
    generated_at = generated_at or datetime.now()
    for scale in SCALES:
        if theme.body_size * scale < theme.min_body_size - 1e-6:
            break
        pdf = _try_one_page(sheet, sources, generated_at, theme, scale)
        if pdf is not None:
            return RenderResult(pdf, 1, scale)
    pdf, pages = _multi_page(sheet, sources, generated_at, theme)
    logger.info("pdf.study_sheet.multi_page pages=%s", pages)
    return RenderResult(pdf, pages, 1.0)
