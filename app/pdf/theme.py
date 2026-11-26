"""Visual theme for generated PDFs: palette, section styles, sizes and fonts.

Change the look here (or pass another ``Theme`` to a renderer) without touching
layout code. Colours are light tints with dark text so sheets print well.
"""
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FontSet:
    """Registered font names, plus a symbol font for characters the main font lacks."""
    regular: str
    bold: str
    symbol: str | None = None
    covered: frozenset[int] | None = None          # code points of the main font; None = unknown
    symbol_covered: frozenset[int] = frozenset()


@dataclass(frozen=True)
class SectionStyle:
    """Heading text and colours of one kind of section."""
    label: str
    accent: str            # heading, heading rule and bullet colour
    tint: str = ""         # background of a callout box; unused for open sections
    callout: bool = False  # tinted box, kept for the recall sections (remember, summary)


def _font_candidates() -> tuple[tuple[str, str], ...]:
    """(regular, bold) TrueType pairs to try, best Unicode coverage first.

    Unicode fonts matter for formulas (Greek letters, ≤, √, ...). ReportLab's
    bundled Vera is the last resort and is always present.
    """
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    import reportlab
    bundled = Path(reportlab.__file__).resolve().parent / "fonts"
    return (
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
        (str(windows / "arial.ttf"), str(windows / "arialbd.ttf")),
        (str(windows / "segoeui.ttf"), str(windows / "segoeuib.ttf")),
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        (str(bundled / "Vera.ttf"), str(bundled / "VeraBd.ttf")),
    )


def _symbol_candidates() -> tuple[str, ...]:
    """Fonts with wide symbol coverage, used per character when the main font lacks one."""
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    return (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        str(windows / "seguisym.ttf"),
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    )


@dataclass(frozen=True)
class Theme:
    """Everything that controls how a generated PDF looks."""
    brand_name: str = "LearnerTutors"
    page_size: tuple[float, float] = A4
    margin: float = 12 * mm
    gutter: float = 5 * mm          # space between columns and between cards

    # Sizes at scale 1.0. Renderers may scale down to fit one page, but body
    # text never goes below min_body_size.
    title_size: float = 19.0
    heading_size: float = 10.5
    body_size: float = 9.4
    lead_size: float = 11.0         # the overview, read first under the title
    min_body_size: float = 8.0
    footer_size: float = 7.0
    corner_radius: float = 1.2 * mm

    # "Fresh study" palette, built with UI Color Palette (OKLCH, Tailwind stops).
    # One core colour (teal) carries structure, one accent (coral) is kept for
    # "Remember this" only; everything else is slate neutrals. Contrast on white:
    # primary 8.8:1, accent 5.0:1, text 16.5:1, muted 6.5:1 (all WCAG AA or better).
    primary: str = "#005454"        # Teal 700: headings, heading rules, bullets
    primary_tint: str = "#EDF5F5"   # very light teal: summary box
    accent: str = "#C14530"         # Coral 500: the one accent
    accent_tint: str = "#FCEEEA"    # very light coral: "Remember this" box
    text: str = "#182026"           # Slate 900
    muted: str = "#545F68"          # Slate 600
    rule: str = "#CED9E2"           # Slate 100: hairlines only, never text

    sections: dict[str, SectionStyle] = field(default_factory=lambda: {
        "overview": SectionStyle("Topic overview", "#005454"),
        "key_concepts": SectionStyle("Key concepts", "#005454"),
        "definitions": SectionStyle("Important definitions", "#005454"),
        "main_points": SectionStyle("Main points", "#005454"),
        "examples": SectionStyle("Examples & applications", "#005454"),
        "formulas": SectionStyle("Formulas & equations", "#005454"),
        "remember": SectionStyle("Remember this", "#C14530", "#FCEEEA", callout=True),
        "summary": SectionStyle("Quick revision summary", "#005454", "#EDF5F5", callout=True),
    })
    font_pairs: tuple[tuple[str, str], ...] = field(default_factory=_font_candidates)
    symbol_fonts: tuple[str, ...] = field(default_factory=_symbol_candidates)

    def color(self, hex_value: str) -> HexColor:
        """A ReportLab colour from a hex string."""
        return HexColor(hex_value)

    def fonts(self) -> FontSet:
        """Registered fonts for this theme."""
        return register_fonts(self.font_pairs, self.symbol_fonts)


def _coverage(font_name: str) -> frozenset[int] | None:
    """Code points a registered TrueType font can draw (None for built-in Type 1 fonts)."""
    from reportlab.pdfbase import pdfmetrics

    face = getattr(pdfmetrics.getFont(font_name), "face", None)
    mapping = getattr(face, "charToGlyph", None)
    return frozenset(mapping) if mapping else None


@lru_cache(maxsize=8)
def register_fonts(pairs: tuple[tuple[str, str], ...],
                   symbols: tuple[str, ...] = ()) -> FontSet:
    """Register the first available pair and symbol font; return them as a FontSet."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular, bold = _register_pair(pairs)
    symbol, symbol_covered = None, frozenset()
    for path in symbols:
        if not Path(path).is_file():
            continue
        try:
            name = "LT-Symbol-" + Path(path).stem
            pdfmetrics.registerFont(TTFont(name, path))
            symbol, symbol_covered = name, _coverage(name) or frozenset()
            break
        except Exception:
            logger.warning("pdf.fonts.symbol_unusable path=%s", path, exc_info=True)
    return FontSet(regular, bold, symbol, _coverage(regular), symbol_covered)


def _register_pair(pairs: tuple[tuple[str, str], ...]) -> tuple[str, str]:
    """Register the first available TrueType pair; fall back to Helvetica.

    Helvetica cannot show non-Latin characters such as Greek letters, so the
    fallback is logged as a warning.
    """
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for regular, bold in pairs:
        if not (Path(regular).is_file() and Path(bold).is_file()):
            continue
        try:
            name = "LT-" + Path(regular).stem
            pdfmetrics.registerFont(TTFont(name, regular))
            pdfmetrics.registerFont(TTFont(name + "-Bold", bold))
            addMapping(name, 0, 0, name)
            addMapping(name, 1, 0, name + "-Bold")
            addMapping(name, 0, 1, name)
            addMapping(name, 1, 1, name + "-Bold")
            logger.info("pdf.fonts regular=%s", Path(regular).name)
            return name, name + "-Bold"
        except Exception:  # a corrupt or unsupported font file: try the next one
            logger.warning("pdf.fonts.unusable path=%s", regular, exc_info=True)
    logger.warning("pdf.fonts.fallback font=Helvetica (non-Latin characters may not display)")
    return "Helvetica", "Helvetica-Bold"


DEFAULT_THEME = Theme()
