"""Render an ``ExamPaper`` as an A4 practice exam paper PDF.

Layout: the shared header (title, date line, teal rule), a paper-details line
(time allowed, total marks, number of questions: only the values that are
known), the past paper's general instructions, a one-line notice that this is
a practice paper and not an official exam, then each section as a ruled
heading with its instructions and marks, followed by its questions. Questions
are numbered continuously; MCQ options are lettered A, B, ...; sub-parts use
the past paper's style. Marks sit in a right-hand column, as on a real paper.
Every page has the shared footer (sources, AI notice, page x of n). The paper
has no answers: ``ExamPaper`` carries none.
"""
import io
import logging
from datetime import datetime

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

from app.pdf import components as ui
from app.pdf.study_sheet import RenderResult, sources_text
from app.pdf.theme import DEFAULT_THEME, Theme
from app.schemas import SourceItem
from app.services.exam_paper import ExamPaper, ExamQuestion, ExamSection

logger = logging.getLogger(__name__)

NUMBER_W = 9 * mm          # "12." column
MARKS_W = 19 * mm          # "[10 marks]" column
_ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")


def _marks(value: float | None) -> str:
    """'[4 marks]', '[1 mark]', '[2.5 marks]'; empty when unknown."""
    if value is None:
        return ""
    number = int(value) if float(value).is_integer() else value
    return f"[{number} mark" + ("" if number == 1 else "s") + "]"


def _duration(minutes: int) -> str:
    """'2 hours', '1 hour 30 minutes', '45 minutes'."""
    hours, rest = divmod(int(minutes), 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour" + ("" if hours == 1 else "s"))
    if rest:
        parts.append(f"{rest} minute" + ("" if rest == 1 else "s"))
    return " ".join(parts)


def _subpart_label(style: str, index: int) -> str:
    """(a), (i) or 1. for the sub-part at ``index``."""
    if style == "i_ii_iii" and index < len(_ROMAN):
        return f"({_ROMAN[index]})"
    if style == "1_2_3":
        return f"{index + 1}."
    return f"({chr(ord('a') + index)})"


class _Styles:
    """Paragraph styles for the paper, derived from the shared styles."""

    def __init__(self, theme: Theme, base: ui.Styles):
        muted = theme.color(theme.muted)
        self.base = base
        self.body = base.body
        self.number = ParagraphStyle("exam-number", parent=base.body, fontName=base.fonts.bold)
        self.marks = ParagraphStyle("exam-marks", parent=base.body, textColor=muted, alignment=2)
        self.option = ParagraphStyle("exam-option", parent=base.body, leftIndent=6 * mm,
                                     firstLineIndent=-6 * mm)
        self.subpart = ParagraphStyle("exam-subpart", parent=base.body, leftIndent=8 * mm,
                                      firstLineIndent=-8 * mm)
        self.note = ParagraphStyle("exam-note", parent=base.body, textColor=muted,
                                   fontSize=base.body.fontSize - 0.6,
                                   leading=(base.body.fontSize - 0.6) * 1.3)
        self.instruction = ParagraphStyle("exam-instruction", parent=base.body, leftIndent=5 * mm,
                                          firstLineIndent=-5 * mm)
        self.heading = ParagraphStyle("exam-heading", parent=base.heading,
                                      textColor=theme.color(theme.primary))
        self.end = ParagraphStyle("exam-end", parent=self.note, alignment=1)


def _t(value: str, styles: _Styles) -> str:
    return ui.rich(value, styles.base.fonts)


def _details(paper: ExamPaper, styles: _Styles) -> list:
    """Time allowed, total marks and question count, then the general instructions."""
    facts = []
    if paper.duration_minutes:
        facts.append(f"<b>Time allowed:</b> {_duration(paper.duration_minutes)}")
    if paper.total_marks is not None:
        total = paper.total_marks
        facts.append(f"<b>Total marks:</b> {int(total) if float(total).is_integer() else total}")
    count = paper.question_count()
    facts.append(f"<b>Questions:</b> {count}")
    flow = [Paragraph("    ·    ".join(facts), styles.body), Spacer(1, 3 * mm)]
    if paper.general_instructions:
        flow.append(Paragraph("<b>Instructions</b>", styles.body))
        flow.append(Spacer(1, 1 * mm))
        flow += [Paragraph(f"{number}.&nbsp;&nbsp;{_t(item, styles)}", styles.instruction)
                 for number, item in enumerate(paper.general_instructions, 1)]
        flow.append(Spacer(1, 3 * mm))
    basis = ("in the format of your lecturer's past papers" if paper.follows_past_papers
             else "in the structure you asked for")
    flow.append(Paragraph(
        f"Practice paper written by LearnerTutors from your course materials, {basis}. "
        "It is not an official exam and has no answer key.", styles.note))
    flow.append(Spacer(1, 5 * mm))
    return flow


def _question_marks(question: ExamQuestion) -> float | None:
    """The question's marks, or the sum of its sub-part marks when only those are known."""
    if question.marks is not None:
        return question.marks
    if question.subparts and all(part.marks for part in question.subparts):
        return sum(part.marks for part in question.subparts)
    return None


def _section_heading(section: ExamSection, theme: Theme, styles: _Styles, width: float) -> Table:
    """Ruled section heading with the section's marks on the right."""
    marks = [_question_marks(question) for question in section.questions]
    total = None
    if all(mark is not None for mark in marks):
        counted = sorted(marks, reverse=True)[:section.answer_count] if section.answer_count else marks
        total = sum(counted)
    row = [Paragraph(_t(section.title, styles), styles.heading),
           Paragraph(_marks(total), styles.marks)]
    rows = [row]
    if section.instructions:
        rows.append([Paragraph(_t(section.instructions, styles), styles.body), ""])
    table = Table(rows, colWidths=[width - MARKS_W - 6 * mm, MARKS_W + 6 * mm])
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, theme.color(theme.primary)),
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
    ]
    if section.instructions:
        style += [("SPAN", (0, 1), (-1, 1)), ("TOPPADDING", (0, 1), (-1, 1), 4)]
    table.setStyle(TableStyle(style))
    return table


def _question(number: int, question: ExamQuestion, section: ExamSection, styles: _Styles,
              width: float) -> Table:
    """One question as a 3-column table: number, text (options/sub-parts), marks."""
    show_question_marks = not question.subparts
    rows = [[Paragraph(f"{number}.", styles.number), Paragraph(_t(question.text, styles), styles.body),
             Paragraph(_marks(question.marks) if show_question_marks else "", styles.marks)]]
    for index, option in enumerate(question.options):
        rows.append(["", Paragraph(f"{chr(ord('A') + index)}.&nbsp;&nbsp;&nbsp;{_t(option, styles)}",
                                   styles.option), ""])
    for index, part in enumerate(question.subparts):
        label = _subpart_label(section.subpart_style, index)
        rows.append(["", Paragraph(f"{label}&nbsp;&nbsp;{_t(part.text, styles)}", styles.subpart),
                     Paragraph(_marks(part.marks), styles.marks)])
    table = Table(rows, colWidths=[NUMBER_W, width - NUMBER_W - MARKS_W, MARKS_W],
                  splitByRow=1)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (1, 0), (1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("TOPPADDING", (0, 1), (-1, -1), 2.5),
    ]))
    return table


def _story(paper: ExamPaper, theme: Theme, styles: _Styles, width: float) -> list:
    """Everything below the header, in reading order."""
    story = [NextPageTemplate("later")] + _details(paper, styles)
    number = 0
    for section in paper.sections:
        # Never leave a section heading alone at the bottom of a page.
        story += [CondPageBreak(40 * mm), _section_heading(section, theme, styles, width),
                  Spacer(1, 3.5 * mm)]
        for question in section.questions:
            number += 1
            story += [CondPageBreak(18 * mm), _question(number, question, section, styles, width),
                      Spacer(1, 4.5 * mm)]
        story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("End of paper", styles.end))
    return story


class _CountingDoc(BaseDocTemplate):
    """Doc template that records how many pages were produced."""
    pages = 0

    def afterPage(self):
        """Count each finished page."""
        self.pages = self.page


def _subtitle(generated_at: datetime) -> str:
    return f"Practice exam paper · Generated {generated_at.day} {generated_at:%B %Y}"


def render_exam_paper(paper: ExamPaper, sources: list[SourceItem],
                      generated_at: datetime | None = None,
                      theme: Theme = DEFAULT_THEME) -> RenderResult:
    """Render ``paper`` to PDF bytes across as many A4 pages as it needs."""
    generated_at = generated_at or datetime.now()
    base = ui.make_styles(theme, 1.0)
    styles = _Styles(theme, base)
    page_w, page_h = theme.page_size
    width = page_w - 2 * theme.margin
    footer = ui.footer_paragraph(sources_text(sources), theme, base)
    footer_h = ui.footer_height(footer, theme, base, width)
    probe = pdf_canvas.Canvas(io.BytesIO(), pagesize=theme.page_size)
    header_h = ui.draw_header(probe, theme, base, paper.title, _subtitle(generated_at),
                              page_h - theme.margin, width)
    bottom = theme.margin + footer_h + theme.gutter

    def build(total: int) -> tuple[bytes, int]:
        buffer = io.BytesIO()
        doc = _CountingDoc(buffer, pagesize=theme.page_size, leftMargin=theme.margin,
                           rightMargin=theme.margin, topMargin=theme.margin, bottomMargin=bottom,
                           title=paper.title, author=theme.brand_name,
                           subject="Practice exam paper", creator=theme.brand_name)

        def first(c, d):
            ui.draw_header(c, theme, base, paper.title, _subtitle(generated_at),
                           page_h - theme.margin, width)
            ui.draw_footer(c, theme, base, footer, d.page, total, width)

        def later(c, d):
            ui.draw_footer(c, theme, base, footer, d.page, total, width)

        first_frame = Frame(theme.margin, bottom, width,
                            page_h - theme.margin - header_h - theme.gutter * 1.4 - bottom,
                            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        later_frame = Frame(theme.margin, bottom, width, page_h - theme.margin - bottom,
                            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        doc.addPageTemplates([PageTemplate("first", [first_frame], onPage=first),
                              PageTemplate("later", [later_frame], onPage=later)])
        doc.build(_story(paper, theme, styles, width))
        return buffer.getvalue(), doc.pages

    _, pages = build(0)
    pdf, pages = build(pages)
    logger.info("pdf.exam_paper.rendered pages=%s questions=%s", pages, paper.question_count())
    return RenderResult(pdf, pages, 1.0)
