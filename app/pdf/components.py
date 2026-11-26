"""Reusable PDF building blocks: text styles, section blocks, header and footer."""
from dataclasses import dataclass
from xml.sax.saxutils import escape

from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Paragraph, Table, TableStyle

from app.pdf.theme import FontSet, SectionStyle, Theme


def text(value: str) -> str:
    """Escape plain text for a ReportLab Paragraph (its markup is XML-like)."""
    return escape(str(value))


_SUBSCRIPTS = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎",
                       "0123456789+-=()"))
_SUPERSCRIPTS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ",
                         "0123456789+-=()n"))


def rich(value: str, fonts: FontSet) -> str:
    """Escape ``value`` and make every character printable with ``fonts``.

    Unicode sub/superscript characters the main font lacks become <sub>/<super>
    markup (any font can draw them); other missing characters are drawn with
    the symbol font when it has them.
    """
    out = []
    for ch in str(value):
        code = ord(ch)
        has = fonts.covered is None or code in fonts.covered or ch.isspace()
        if ch in _SUBSCRIPTS and not has:
            out.append(f"<sub>{escape(_SUBSCRIPTS[ch])}</sub>")
        elif ch in _SUPERSCRIPTS and not has:
            out.append(f"<super>{escape(_SUPERSCRIPTS[ch])}</super>")
        elif not has and fonts.symbol and code in fonts.symbol_covered:
            out.append(f'<font name="{fonts.symbol}">{escape(ch)}</font>')
        else:
            out.append(escape(ch))
    return "".join(out).replace("</sub><sub>", "").replace("</super><super>", "")


@dataclass(frozen=True)
class Styles:
    """Paragraph styles for one theme at one scale."""
    title: ParagraphStyle
    subtitle: ParagraphStyle
    heading: ParagraphStyle
    lead: ParagraphStyle
    body: ParagraphStyle
    bullet: ParagraphStyle
    footer: ParagraphStyle
    scale: float
    fonts: FontSet


def make_styles(theme: Theme, scale: float = 1.0) -> Styles:
    """Build the styles; body text is never smaller than ``theme.min_body_size``."""
    fonts = theme.fonts()
    regular, bold = fonts.regular, fonts.bold
    body_size = max(theme.min_body_size, theme.body_size * scale)
    lead = body_size * 1.28
    body = ParagraphStyle("body", fontName=regular, fontSize=body_size, leading=lead,
                          textColor=theme.color(theme.text))
    return Styles(
        title=ParagraphStyle("title", fontName=bold, fontSize=theme.title_size * scale,
                             leading=theme.title_size * scale * 1.15,
                             textColor=theme.color(theme.text)),
        subtitle=ParagraphStyle("subtitle", fontName=regular, fontSize=max(theme.min_body_size, 8.5 * scale),
                                leading=max(theme.min_body_size, 8.5 * scale) * 1.3,
                                textColor=theme.color(theme.muted)),
        heading=ParagraphStyle("heading", fontName=bold,
                               fontSize=max(theme.min_body_size + 0.5, theme.heading_size * scale),
                               leading=max(theme.min_body_size + 0.5, theme.heading_size * scale) * 1.2),
        lead=ParagraphStyle("lead", parent=body,
                            fontSize=max(body_size, theme.lead_size * scale),
                            leading=max(body_size, theme.lead_size * scale) * 1.38),
        body=body,
        bullet=ParagraphStyle("bullet", parent=body, leftIndent=body_size * 1.05,
                              bulletIndent=0, bulletFontName=bold),
        footer=ParagraphStyle("footer", fontName=regular, fontSize=theme.footer_size,
                              leading=theme.footer_size * 1.35,
                              textColor=theme.color(theme.muted)),
        scale=scale,
        fonts=fonts,
    )


def section_card(section: SectionStyle, rows: list, width: float, theme: Theme,
                 styles: Styles) -> Table:
    """A section: coloured heading on a ruled line, then one table row per item.

    Open sections have no box, so whitespace and the heading rule separate them
    (the ruled heading is the sheet's repeated motif, like a notebook margin).
    Callout sections (remember, summary) sit in a tinted box because they hold
    the points a student should re-read last.

    Rows are separate table rows so a section can split across pages when a
    document needs more than one page.
    """
    s = styles.scale
    accent = theme.color(section.accent)
    heading_style = ParagraphStyle("card-heading", parent=styles.heading, textColor=accent)
    heading = Paragraph(text(section.label), heading_style)
    pad = 6.0 * s + 1 if section.callout else 0
    data = [[heading]] + [[row] for row in rows]
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.9 if not section.callout else 0.6, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), pad), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6 * s), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6 * s),
        ("TOPPADDING", (0, 0), (-1, 0), pad * 0.8), ("BOTTOMPADDING", (0, 0), (-1, 0), 3.0 * s),
        ("TOPPADDING", (0, 1), (-1, 1), 4.5 * s),
        ("BOTTOMPADDING", (0, -1), (-1, -1), pad * 0.8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if section.callout and section.tint:
        style += [("BACKGROUND", (0, 0), (-1, -1), theme.color(section.tint)),
                  ("ROUNDEDCORNERS", [theme.corner_radius] * 4)]
    return Table(data, colWidths=[width], style=TableStyle(style), splitByRow=1, repeatRows=1)


def accent_bullet_style(section: SectionStyle, theme: Theme, styles: Styles) -> ParagraphStyle:
    """Bullet style whose bullet glyph uses the section accent colour."""
    return ParagraphStyle(f"bullet-{section.label}", parent=styles.bullet,
                          bulletColor=theme.color(section.accent))


def draw_header(canvas, theme: Theme, styles: Styles, title: str, subtitle: str,
                top: float, width: float) -> float:
    """Draw the header at ``top``: title, date line, brand, and a teal rule; return its height.

    No band or box: the title is the page's focal point and the heavy rule
    under it echoes the ruled section headings below.
    """
    x = theme.margin
    title_par = Paragraph(rich(title, styles.fonts), styles.title)
    sub_par = Paragraph(text(subtitle), styles.subtitle)
    brand_style = ParagraphStyle("brand", parent=styles.subtitle, fontName=styles.fonts.bold,
                                 textColor=theme.color(theme.primary), alignment=2)
    brand = Paragraph(text(theme.brand_name), brand_style)
    brand_w = stringWidth(theme.brand_name, styles.fonts.bold, brand_style.fontSize) + 6
    inner_w = width - brand_w - 8
    _, th = title_par.wrap(inner_w, 200)
    _, sh = sub_par.wrap(inner_w, 50)
    rule_gap, rule_w = 5.0 * styles.scale, 1.8
    height = th + 2 + sh + rule_gap + rule_w
    title_par.drawOn(canvas, x, top - th)
    sub_par.drawOn(canvas, x, top - th - 2 - sh)
    _, bh = brand.wrap(brand_w, 30)
    brand.drawOn(canvas, x + width - brand_w, top - th - 2 - sh + (sh - bh) / 2)
    canvas.saveState()
    canvas.setStrokeColor(theme.color(theme.primary))
    canvas.setLineWidth(rule_w)
    rule_y = top - height + rule_w / 2
    canvas.line(x, rule_y, x + width, rule_y)
    canvas.restoreState()
    return height


_PAGE_LABEL_W = 60  # room reserved on the right for "Page x of n"


def _brand_line(theme: Theme, styles: Styles) -> Paragraph:
    """Branding and the AI-generated notice."""
    return Paragraph(
        f"<b>{text(theme.brand_name)}</b> · AI-generated from your course materials. "
        "Check important details against the original documents.", styles.footer)


def footer_paragraph(sources_text: str, theme: Theme, styles: Styles) -> Paragraph:
    """The wrapped sources line shown in the footer."""
    return Paragraph(sources_text, styles.footer)


def draw_footer(canvas, theme: Theme, styles: Styles, sources: Paragraph, page: int,
                pages: int, width: float) -> None:
    """Draw sources, page number and branding at the bottom margin."""
    x, bottom = theme.margin, theme.margin
    brand = _brand_line(theme, styles)
    _, bh = brand.wrap(width - _PAGE_LABEL_W, 40)
    _, sh = sources.wrap(width - _PAGE_LABEL_W, 200)
    brand.drawOn(canvas, x, bottom)
    sources.drawOn(canvas, x, bottom + bh + 1)
    canvas.saveState()
    canvas.setFont(styles.footer.fontName, theme.footer_size)
    canvas.setFillColor(theme.color(theme.muted))
    canvas.drawRightString(x + width, bottom, f"Page {page} of {pages}")
    canvas.setStrokeColor(theme.color(theme.rule))
    canvas.setLineWidth(0.6)
    rule_y = bottom + bh + sh + 4
    canvas.line(x, rule_y, x + width, rule_y)
    canvas.restoreState()


def footer_height(sources: Paragraph, theme: Theme, styles: Styles, width: float) -> float:
    """Height the footer needs, including its rule."""
    _, bh = _brand_line(theme, styles).wrap(width - _PAGE_LABEL_W, 40)
    _, sh = sources.wrap(width - _PAGE_LABEL_W, 200)
    return bh + sh + 6
