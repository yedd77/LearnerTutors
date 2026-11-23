"""Study sheet PDF rendering: one-page fit, readable sizes, nothing clipped or dropped."""
from datetime import datetime

import pymupdf
import pytest

from app.pdf.study_sheet import render_study_sheet, sources_text
from app.pdf.theme import DEFAULT_THEME
from app.schemas import SourceItem
from app.services.study_sheet import StudySheet

SOURCES = [SourceItem(document_id="d1", filename="Week3.pdf", page=p, chunk_id=f"c{p}",
                      excerpt="", markers=["S1"]) for p in (5, 2)]


def sheet(**overrides) -> StudySheet:
    data = dict(
        title="Cell Respiration", coverage="sufficient",
        overview="Cells release energy from glucose to make ATP.",
        key_concepts=["Glycolysis happens in the cytoplasm", "The Krebs cycle runs in mitochondria"],
        definitions=[{"term": "ATP", "meaning": "The cell's energy currency"}],
        main_points=["Aerobic respiration needs oxygen"],
        summary=["Glucose + oxygen gives energy, CO₂ and water"],
    )
    data.update(overrides)
    return StudySheet(**data)


def pdf_doc(result):
    return pymupdf.open(stream=result.pdf, filetype="pdf")


def all_text(doc) -> str:
    return " ".join(" ".join(page.get_text().split()) for page in doc)


def font_sizes(doc) -> list[float]:
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                sizes += [span["size"] for span in line["spans"] if span["text"].strip()]
    return sizes


def test_typical_sheet_is_one_a4_page_with_header_and_footer():
    result = render_study_sheet(sheet(), SOURCES, datetime(2026, 11, 20))
    doc = pdf_doc(result)
    assert result.pages == doc.page_count == 1
    width, height = doc[0].rect.width, doc[0].rect.height
    assert (round(width), round(height)) == (595, 842)   # A4 portrait in points
    text = all_text(doc)
    assert "Cell Respiration" in text and "Generated 20 November 2026" in text
    assert "Sources: Week3.pdf (p. 2, 5)" in text and "Page 1 of 1" in text
    assert doc.metadata["title"] == "Cell Respiration" and doc.metadata["author"] == "LearnerTutors"


def test_only_sections_with_content_appear():
    text = all_text(pdf_doc(render_study_sheet(sheet(), SOURCES)))
    assert "Key concepts" in text and "Important definitions" in text
    for absent in ("Formulas & equations", "Examples & applications", "Remember this"):
        assert absent not in text


def test_short_sheets_grow_and_body_text_is_never_below_the_minimum():
    result = render_study_sheet(sheet(), SOURCES)
    assert result.scale > 1.0
    long = sheet(main_points=[f"Point {i}: " + "aerobic respiration releases energy " * 6
                              for i in range(10)])
    for res in (result, render_study_sheet(long, SOURCES)):
        sizes = font_sizes(pdf_doc(res))
        # Footer text is the only text allowed below body size.
        assert min(s for s in sizes if s > DEFAULT_THEME.footer_size + 0.1) >= DEFAULT_THEME.min_body_size - 0.01


def test_too_much_content_flows_onto_more_pages_without_losing_text():
    items = [f"Item {i:02d} unique-marker-{i:02d} " + "the electron transport chain pumps protons " * 4
             for i in range(12)]
    big = sheet(key_concepts=items, main_points=items, examples=items[:8])
    result = render_study_sheet(big, SOURCES)
    doc = pdf_doc(result)
    assert result.pages == doc.page_count > 1
    text = all_text(doc)
    for i in range(12):
        assert f"unique-marker-{i:02d}" in text      # nothing clipped or dropped
    assert f"Page {doc.page_count} of {doc.page_count}" in text


def test_markup_characters_and_formula_symbols_are_rendered_literally():
    tricky = sheet(title="Rates & <Ratios>",
                   formulas=[{"expression": "a < b & c > d", "meaning": "x² + y₂ ≤ √2"}])
    text = all_text(pdf_doc(render_study_sheet(tricky, SOURCES)))
    assert "Rates & <Ratios>" in text and "a < b & c > d" in text
    assert "y2" in text and "≤" in text


@pytest.mark.parametrize("count,expected", [(0, "course materials"), (10, "and 2 more documents")])
def test_sources_line(count, expected):
    sources = [SourceItem(document_id=str(i), filename=f"f{i}.pdf", page=None, chunk_id=str(i),
                          excerpt="", markers=[]) for i in range(count)]
    assert expected in sources_text(sources)
