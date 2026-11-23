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
