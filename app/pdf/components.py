"""Reusable PDF building blocks: text styles, vector icons, section cards, header and footer."""
import math
from dataclasses import dataclass
from xml.sax.saxutils import escape

from reportlab.lib.colors import white
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Flowable, Paragraph, Table, TableStyle

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
                             textColor=theme.color(theme.primary)),
        subtitle=ParagraphStyle("subtitle", fontName=regular, fontSize=max(theme.min_body_size, 8.5 * scale),
                                leading=max(theme.min_body_size, 8.5 * scale) * 1.3,
                                textColor=theme.color(theme.muted)),
        heading=ParagraphStyle("heading", fontName=bold,
                               fontSize=max(theme.min_body_size + 0.5, theme.heading_size * scale),
                               leading=max(theme.min_body_size + 0.5, theme.heading_size * scale) * 1.2),
        body=body,
        bullet=ParagraphStyle("bullet", parent=body, leftIndent=body_size * 1.05,
                              bulletIndent=0, bulletFontName=bold),
        footer=ParagraphStyle("footer", fontName=regular, fontSize=theme.footer_size,
                              leading=theme.footer_size * 1.35,
                              textColor=theme.color(theme.muted)),
        scale=scale,
        fonts=fonts,
    )


class Icon(Flowable):
    """A small filled circle with a white vector glyph (no emoji fonts needed)."""

    def __init__(self, kind: str, color, size: float):
        super().__init__()
        self.kind, self.color, self.size = kind, color, size
        self.width = self.height = size

    def draw(self) -> None:
        """Draw the badge and its glyph."""
        c, s = self.canv, self.size
        r = s / 2
        c.saveState()
        c.setFillColor(self.color)
        c.circle(r, r, r, stroke=0, fill=1)
        c.setStrokeColor(white)
        c.setFillColor(white)
        c.setLineWidth(max(0.8, s * 0.09))
        c.setLineCap(1)
        c.setLineJoin(1)
        getattr(self, "_" + self.kind, self._dot)(c, s)
        c.restoreState()

    @staticmethod
    def _dot(c, s):
        c.circle(s / 2, s / 2, s * 0.12, stroke=0, fill=1)

    @staticmethod
    def _info(c, s):
        c.circle(s / 2, s * 0.72, s * 0.07, stroke=0, fill=1)
        c.line(s / 2, s * 0.24, s / 2, s * 0.55)

    @staticmethod
    def _star(c, s):
        path = c.beginPath()
        for i in range(10):
            radius = s * (0.33 if i % 2 == 0 else 0.14)
            angle = math.pi / 2 + i * math.pi / 5
            point = (s / 2 + radius * math.cos(angle), s / 2 + radius * math.sin(angle) - s * 0.02)
            path.moveTo(*point) if i == 0 else path.lineTo(*point)
        path.close()
        c.drawPath(path, stroke=0, fill=1)

    @staticmethod
    def _book(c, s):
        c.setLineWidth(max(0.6, s * 0.07))
        c.line(s / 2, s * 0.27, s / 2, s * 0.72)
        c.lines([(s * 0.22, s * 0.70, s / 2, s * 0.72), (s * 0.22, s * 0.70, s * 0.22, s * 0.28),
                 (s * 0.22, s * 0.28, s / 2, s * 0.27), (s * 0.78, s * 0.70, s / 2, s * 0.72),
                 (s * 0.78, s * 0.70, s * 0.78, s * 0.28), (s * 0.78, s * 0.28, s / 2, s * 0.27)])

    @staticmethod
    def _list(c, s):
        for y in (0.30, 0.50, 0.70):
            c.circle(s * 0.27, s * y, s * 0.045, stroke=0, fill=1)
            c.line(s * 0.40, s * y, s * 0.76, s * y)

    @staticmethod
    def _arrow(c, s):
        c.line(s * 0.24, s / 2, s * 0.74, s / 2)
        c.lines([(s * 0.56, s * 0.68, s * 0.76, s / 2), (s * 0.56, s * 0.32, s * 0.76, s / 2)])

    @staticmethod
    def _formula(c, s):
        # Sigma-like glyph drawn as lines, so no font is needed.
        c.lines([(s * 0.70, s * 0.72, s * 0.30, s * 0.72), (s * 0.30, s * 0.72, s * 0.52, s / 2),
                 (s * 0.52, s / 2, s * 0.30, s * 0.28), (s * 0.30, s * 0.28, s * 0.70, s * 0.28)])

    @staticmethod
    def _alert(c, s):
        c.line(s / 2, s * 0.42, s / 2, s * 0.74)
        c.circle(s / 2, s * 0.27, s * 0.065, stroke=0, fill=1)

    @staticmethod
    def _check(c, s):
        c.lines([(s * 0.26, s * 0.50, s * 0.43, s * 0.33), (s * 0.43, s * 0.33, s * 0.75, s * 0.68)])


def section_card(section: SectionStyle, rows: list, width: float, theme: Theme,
                 styles: Styles) -> Table:
    """A rounded card: icon + coloured heading, then one table row per item.

    Rows are separate table rows so a card can split across pages when a
    document needs more than one page.
    """
    s = styles.scale
    accent, tint = theme.color(section.accent), theme.color(section.tint)
    icon_size = max(10.0, 13.0 * s)
    heading_style = ParagraphStyle("card-heading", parent=styles.heading, textColor=accent)
    heading = Table(
        [[Icon(section.icon, accent, icon_size), Paragraph(text(section.label), heading_style)]],
        colWidths=[icon_size + 5, None],
        style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                          ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                          ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]),
    )
    pad = 7.0 * s + 1
    data = [[heading]] + [[row] for row in rows]
    style = [
        ("BACKGROUND", (0, 0), (-1, -1), tint),
        ("ROUNDEDCORNERS", [theme.corner_radius] * 4),
        ("BOX", (0, 0), (-1, -1), 1.1 if section.callout else 0.5, accent if section.callout
         else theme.color(theme.rule)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), pad), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6 * s), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6 * s),
        ("TOPPADDING", (0, 0), (-1, 0), pad * 0.75), ("BOTTOMPADDING", (0, 0), (-1, 0), 3.5 * s),
        ("TOPPADDING", (0, 1), (-1, 1), 4 * s),
        ("BOTTOMPADDING", (0, -1), (-1, -1), pad * 0.8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    return Table(data, colWidths=[width], style=TableStyle(style), splitByRow=1, repeatRows=1)


def accent_bullet_style(section: SectionStyle, theme: Theme, styles: Styles) -> ParagraphStyle:
    """Bullet style whose bullet glyph uses the section accent colour."""
    return ParagraphStyle(f"bullet-{section.label}", parent=styles.bullet,
                          bulletColor=theme.color(section.accent))


def draw_header(canvas, theme: Theme, styles: Styles, title: str, subtitle: str,
                top: float, width: float) -> float:
    """Draw the tinted header band at ``top``; return its height."""
    x = theme.margin
    title_par = Paragraph(rich(title, styles.fonts), styles.title)
    sub_par = Paragraph(text(subtitle), styles.subtitle)
    brand_style = ParagraphStyle("brand", parent=styles.subtitle, fontName=styles.fonts.bold,
                                 textColor=theme.color(theme.primary), alignment=2)
    brand = Paragraph(text(theme.brand_name), brand_style)
    pad = 5.0 * styles.scale + 2
    brand_w = stringWidth(theme.brand_name, styles.fonts.bold, brand_style.fontSize) + 6
    inner_w = width - 2 * pad - brand_w
    _, th = title_par.wrap(inner_w, 200)
    _, sh = sub_par.wrap(inner_w, 50)
    height = pad + th + 2 + sh + pad
    canvas.saveState()
    canvas.setFillColor(theme.color(theme.primary_tint))
    canvas.roundRect(x, top - height, width, height, theme.corner_radius, stroke=0, fill=1)
    canvas.setFillColor(theme.color(theme.primary))
    canvas.rect(x, top - height + theme.corner_radius, 2.2, height - 2 * theme.corner_radius,
                stroke=0, fill=1)
    canvas.restoreState()
    title_par.drawOn(canvas, x + pad + 3, top - pad - th)
    sub_par.drawOn(canvas, x + pad + 3, top - pad - th - 2 - sh)
    _, bh = brand.wrap(brand_w, 30)
    brand.drawOn(canvas, x + width - pad - brand_w, top - pad - bh)
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
