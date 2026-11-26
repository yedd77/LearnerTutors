"""Exam paper PDF rendering: layout facts, page flow, literal text, no answers."""
from datetime import datetime

import pymupdf

from app.pdf.exam_paper import render_exam_paper
from app.schemas import SourceItem
from app.services.exam_paper import ExamPaper

SOURCES = [SourceItem(document_id="d1", filename="Week3.pdf", page=p, chunk_id=f"c{p}",
                      excerpt="", markers=["S1"]) for p in (5, 2)]


def paper(questions: int = 3, **overrides) -> ExamPaper:
    data = {"title": "Practice Exam: Cells", "duration_minutes": 90, "total_marks": questions,
            "general_instructions": ["Answer ALL questions."],
            "sections": [{"title": "Section A", "instructions": "Each question carries 1 mark.",
                          "question_type": "mcq",
                          "questions": [{"text": f"unique-question-{i:02d} about cells?", "marks": 1,
                                         "options": ["One", "Two", "Three", "Four"]}
                                        for i in range(questions)]}]}
    data.update(overrides)
    return ExamPaper.model_validate(data)


def text_of(result) -> str:
    doc = pymupdf.open(stream=result.pdf, filetype="pdf")
    return " ".join(" ".join(page.get_text().split()) for page in doc), doc


def test_header_details_and_footer():
    result = render_exam_paper(paper(), SOURCES, datetime(2026, 11, 26))
    text, doc = text_of(result)
    assert result.pages == doc.page_count == 1
    assert "Practice Exam: Cells" in text and "Practice exam paper · Generated 26 November 2026" in text
    assert "Time allowed: 1 hour 30 minutes" in text and "Total marks: 3" in text
    assert "Questions: 3" in text and "Answer ALL questions." in text
    assert "A. One" in text and "[1 mark]" in text and "End of paper" in text
    assert "Sources: Week3.pdf (p. 2, 5)" in text and "Page 1 of 1" in text
    assert doc.metadata["title"] == "Practice Exam: Cells"


def test_unknown_duration_and_marks_are_left_out():
    result = render_exam_paper(paper(duration_minutes=None, total_marks=None), SOURCES)
    text, _ = text_of(result)
    assert "Time allowed" not in text and "Total marks" not in text


def test_long_papers_flow_onto_more_pages_without_losing_questions():
    result = render_exam_paper(paper(questions=40), SOURCES)
    text, doc = text_of(result)
    assert result.pages == doc.page_count > 1
    for i in range(40):
        assert f"unique-question-{i:02d}" in text
    assert f"Page {doc.page_count} of {doc.page_count}" in text


def test_markup_characters_and_subparts_render_literally():
    structured = paper(sections=[{"title": "Section B", "question_type": "structured",
                                  "subpart_style": "i_ii_iii", "questions": [{
                                      "text": "Rates & <Ratios> when a < b", "subparts": [
                                          {"text": "Find x² ≤ 4", "marks": 3},
                                          {"text": "Explain.", "marks": 2}]}]}])
    text, _ = text_of(render_exam_paper(structured, SOURCES))
    assert "Rates & <Ratios> when a < b" in text and "≤" in text
    assert "(i)" in text and "(ii)" in text and "[3 marks]" in text and "[5 marks]" in text
