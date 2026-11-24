from io import BytesIO

import fitz
import pytest
from docx import Document
from pptx import Presentation
from pptx.util import Inches

from app.services.document_parser import DocumentParseError, extract_sections


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    doc = Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    if table:
        grid = doc.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                grid.cell(r, c).text = value
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def make_pptx(slides: list[str]) -> bytes:
    presentation = Presentation()
    for text in slides:
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        if text:
            box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
            box.text_frame.text = text
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("name", ["notes.txt", "notes.md", "NOTES.TXT"])
def test_text_and_markdown(name):
    sections = extract_sections(name, "# Title\n\nHello café".encode("utf-8"))
    expected = {"page": None, "text": "# Title\n\nHello café"}
    if name.lower().endswith(".md"):
        expected["heading"] = "Title"     # Markdown headings are kept for chunk context
    assert sections == [expected]


def test_utf8_bom_is_removed():
    sections = extract_sections("a.txt", b"\xef\xbb\xbfhello")
    assert sections[0]["text"] == "hello"


def test_pdf_pages_are_numbered_and_blank_pages_skipped():
    sections = extract_sections("a.pdf", make_pdf(["First page text", "", "Third page text"]))
    assert [(s["page"], s["text"]) for s in sections] == [
        (1, "First page text"), (3, "Third page text"),
    ]


def test_pdf_page_limit(configure):
    configure(MAX_PAGES="2")
    with pytest.raises(DocumentParseError, match="2-page limit"):
        extract_sections("a.pdf", make_pdf(["one", "two", "three"]))


def test_pdf_with_no_text_reports_ocr_hint():
    with pytest.raises(DocumentParseError, match="No extractable text"):
        extract_sections("scan.pdf", make_pdf(["", ""]))


def test_docx_paragraphs_and_tables():
    sections = extract_sections(
        "a.docx", make_docx(["Intro paragraph", "Second"], table=[["Term", "Meaning"], ["ATP", "Energy"]]),
    )
    assert len(sections) == 1 and sections[0]["page"] is None
    text = sections[0]["text"]
    assert "Intro paragraph" in text and "Second" in text
    assert "Term | Meaning" in text and "ATP | Energy" in text


def test_pptx_slides_are_numbered_and_empty_slides_skipped():
    sections = extract_sections("a.pptx", make_pptx(["Slide one", "", "Slide three"]))
    assert [(s["page"], s["text"]) for s in sections] == [(1, "Slide one"), (3, "Slide three")]


@pytest.mark.parametrize("name,content", [
    ("empty.txt", b""),
    ("blank.txt", b"   \n\t "),
    ("empty.docx", make_docx([])),
])
def test_files_without_text_are_rejected(name, content):
    with pytest.raises(DocumentParseError, match="No extractable text"):
        extract_sections(name, content)


@pytest.mark.parametrize("name", ["virus.exe", "image.png", "noextension", "archive.zip"])
def test_unsupported_extension(name):
    with pytest.raises(DocumentParseError, match="Supported formats"):
        extract_sections(name, b"data")


@pytest.mark.parametrize("name", ["bad.pdf", "bad.docx", "bad.pptx"])
def test_corrupt_files_become_parse_errors(name):
    with pytest.raises(DocumentParseError, match="Could not parse"):
        extract_sections(name, b"this is not a real document")


# --------------------------------------------------------------------------- vector-drawn PDF figures
PLASMA_CAPTION = "FIGURE 1 Plasma concentration versus time"


def _png(width=200, height=150) -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buffer, "PNG")
    return buffer.getvalue()


def _draw_chart(page, top=120.0, left=100.0, width=300.0, height=160.0):
    """Axes, a curve and labels: a typical drawn (not embedded) textbook chart."""
    shape = page.new_shape()
    shape.draw_line((left, top + height), (left + width, top + height))       # x axis
    shape.draw_line((left, top), (left, top + height))                        # y axis
    shape.draw_polyline([(left + 10 + i * (width - 20) / 9, top + height - 10 - (i * (9 - i)) * 3.5)
                         for i in range(10)])                                 # curve
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()
    page.insert_text((left + width / 2 - 10, top + height + 14), "Time", fontsize=8)
    page.insert_text((left - 40, top + 10), "Conc.", fontsize=8)
    return fitz.Rect(left, top, left + width, top + height)


def _page_with(draw) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Drug levels are measured after each dose given to the patient.",
                     fontsize=11)
    draw(page)
    page.insert_text((72, 700), "The concentration then falls as the drug is eliminated from the body.",
                     fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def test_vector_chart_above_a_figure_caption_is_rendered():
    region = {}

    def draw(page):
        region["chart"] = _draw_chart(page)
        page.insert_text((72, 320), PLASMA_CAPTION, fontsize=10)

    [section] = extract_sections("a.pdf", _page_with(draw))
    [image] = section["images"]
    assert image["caption"] == PLASMA_CAPTION and image["page"] == 1
    from PIL import Image

    with Image.open(BytesIO(image["data"])) as rendered:
        assert rendered.format == "PNG"
        chart = region["chart"]
        assert rendered.width >= 2 * chart.width and rendered.height >= 2 * chart.height
    # The image sits just before the caption; label text stays in the page text.
    assert section["text"][image["offset"]:].lstrip().startswith(PLASMA_CAPTION)
    assert "Time" in section["text"]


def test_raster_figure_with_the_same_caption_is_not_duplicated():
    def draw(page):
        page.insert_image(fitz.Rect(100, 120, 400, 280), stream=_png())
        page.insert_text((72, 300), PLASMA_CAPTION, fontsize=10)

    [section] = extract_sections("a.pdf", _page_with(draw))
    [image] = section["images"]
    assert image["caption"] == PLASMA_CAPTION


def test_raster_inside_a_drawn_figure_is_replaced_by_the_rendered_region():
    def draw(page):
        _draw_chart(page)
        page.insert_image(fitz.Rect(150, 140, 300, 250), stream=_png())    # picture inside the axes
        page.insert_text((72, 320), PLASMA_CAPTION, fontsize=10)

    [section] = extract_sections("a.pdf", _page_with(draw))
    [image] = section["images"]
    from PIL import Image

    with Image.open(BytesIO(image["data"])) as rendered:
        assert rendered.width >= 600                     # the whole chart, not the 150 pt picture


def test_table_rules_are_not_rendered():
    def draw(page):
        shape = page.new_shape()
        for y in (120, 150, 180, 210, 240):
            shape.draw_line((100, y), (400, y))
        for x in (100, 250, 400):
            shape.draw_line((x, 120), (x, 240))
        shape.finish(color=(0, 0, 0), width=0.5)
        shape.commit()
        page.insert_text((72, 260), "TABLE 1 Dosing intervals by drug", fontsize=10)

    [section] = extract_sections("a.pdf", _page_with(draw))
    assert "images" not in section


def test_page_rules_alone_are_not_a_figure():
    def draw(page):
        shape = page.new_shape()
        shape.draw_line((20, 40), (page.rect.width - 20, 40))          # header rule
        shape.draw_line((20, 800), (page.rect.width - 20, 800))        # footer rule
        shape.finish(color=(0, 0, 0), width=0.5)
        shape.commit()
        page.insert_text((72, 320), "Figure 2 Loading dose and maintenance dose", fontsize=10)

    [section] = extract_sections("a.pdf", _page_with(draw))
    assert "images" not in section


def test_body_text_paragraph_bounds_the_figure():
    def draw(page):
        shape = page.new_shape()
        shape.draw_line((100, 90), (400, 90))             # an unrelated rule above the paragraph
        shape.draw_line((100, 95), (400, 95))
        shape.finish(color=(0, 0, 0), width=1)
        shape.commit()
        page.insert_text((100, 110), "This is a long body text paragraph line that separates the "
                                     "figure from anything above it.", fontsize=9)
        _draw_chart(page, top=130)
        page.insert_text((72, 330), PLASMA_CAPTION, fontsize=10)

    [section] = extract_sections("a.pdf", _page_with(draw))
    from PIL import Image

    with Image.open(BytesIO(section["images"][0]["data"])) as rendered:
        assert rendered.height < 2 * (330 - 100)          # did not grow over the paragraph


def test_rendered_figure_is_stored_and_anchored_at_its_caption():
    from app.services import document_images
    from app.services.chunker import chunk_sections

    def draw(page):
        _draw_chart(page)
        page.insert_text((72, 320), PLASMA_CAPTION, fontsize=10)

    sections = extract_sections("a.pdf", _page_with(draw))
    chunks = chunk_sections(sections)
    assert document_images.store_document_images(1, "pk", sections, chunks) == 1
    [entry] = document_images._manifest(1, "pk")
    assert entry["caption"] == PLASMA_CAPTION and entry["page"] == 1
    anchor = chunks[entry["anchor"]] if entry["anchor"] >= 0 else chunks[0]
    assert "Plasma concentration" in anchor["text"] or "Time" in anchor["text"]


def test_short_caption_line_still_renders_the_full_width_figure():
    """A caption line like "FIGURE 3" alone is narrower than its figure."""
    def draw(page):
        page.insert_text((72, 100), "This body paragraph spans the whole text column from the left "
                                    "margin to the right margin.", fontsize=9)
        _draw_chart(page, left=330, width=200)            # chart to the right of the caption
        page.insert_text((72, 320), "FIGURE 3", fontsize=10)

    [section] = extract_sections("a.pdf", _page_with(draw))
    from PIL import Image

    with Image.open(BytesIO(section["images"][0]["data"])) as rendered:
        assert rendered.width >= 2 * 200


def test_rotated_pages_are_not_rendered():
    def draw(page):
        _draw_chart(page)
        page.insert_text((72, 320), PLASMA_CAPTION, fontsize=10)

    doc = fitz.open(stream=_page_with(draw), filetype="pdf")
    doc[0].set_rotation(90)
    [section] = extract_sections("a.pdf", doc.tobytes())
    assert "images" not in section
