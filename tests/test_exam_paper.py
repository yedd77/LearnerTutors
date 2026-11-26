"""/exam paper drafts: checked against the past-paper structure, answers never kept."""
import pytest

from app.services import exam_blueprint as eb
from app.services import exam_paper as ep

PLAN = eb.ExamBlueprint.model_validate({
    "sections": [
        {"title": "Section A", "instructions": "Answer ALL questions.", "question_type": "mcq",
         "question_count": 4, "marks_per_question": 1, "options_per_question": 4},
        {"title": "Section B", "instructions": "Answer ONE question.", "question_type": "structured",
         "question_count": 2, "answer_count": 1, "marks_per_question": 10,
         "subparts_per_question": 2, "subpart_style": "a_b_c"},
    ],
    "duration_minutes": 60,
    "general_instructions": ["Write your answers in the booklet provided."],
})
MATCHES = [{"text": "...", "metadata": {"filename": "Week1.pdf", "page": 3}},
           {"text": "...", "metadata": {"filename": "Week2.pdf", "page": 7}}]


def mcq(n: int, **overrides) -> dict:
    data = {"text": f"Which statement about topic {n} is correct?", "marks": 1,
            "options": [f"Option {n}-{i}" for i in range(4)], "answer_index": n % 4,
            "answer": "", "source": "S1"}
    data.update(overrides)
    return data


def structured(n: int, **overrides) -> dict:
    data = {"text": f"A scenario about process {n}.", "marks": 10, "answer": "Key points ...",
            "subparts": [{"text": f"Describe step {n}.", "marks": 4},
                         {"text": f"Explain why step {n} matters.", "marks": 6}],
            "source": "S2"}
    data.update(overrides)
    return data


def draft(**overrides) -> ep.ExamDraft:
    data = {"title": "Practice Exam: Cells", "coverage": "sufficient", "custom_structure": False,
            "sections": [
                {"title": "anything", "instructions": "ignored", "question_type": "mcq",
                 "questions": [mcq(i) for i in range(4)]},
                {"title": "B", "question_type": "structured",
                 "questions": [structured(i) for i in range(2)]},
            ]}
    data.update(overrides)
    return ep.ExamDraft.model_validate(data)


def test_valid_draft_follows_the_blueprint_and_drops_answers():
    paper = ep.build_paper(draft(), PLAN, MATCHES, "cells")
    assert [s.title for s in paper.sections] == ["Section A", "Section B"]
    assert paper.sections[0].instructions == "Answer ALL questions."      # from the past paper
    assert paper.sections[1].answer_count == 1 and paper.sections[1].subpart_style == "a_b_c"
    assert paper.duration_minutes == 60 and paper.total_marks == 4 + 10
    assert paper.general_instructions == ["Write your answers in the booklet provided."]
    assert paper.question_count() == 6 and paper.follows_past_papers
    assert paper.sections[0].questions[0].source.filename == "Week1.pdf"
    dumped = paper.model_dump_json()
    assert "answer_index" not in dumped and "Key points" not in dumped and '"answer"' not in dumped


def test_blueprint_marks_override_the_models_marks():
    sections = draft().model_dump()["sections"]
    sections[0]["questions"][0]["marks"] = 5
    paper = ep.build_paper(draft(sections=sections), PLAN, MATCHES, "cells")
    assert paper.sections[0].questions[0].marks == 1


@pytest.mark.parametrize("mutate, problem", [
    (lambda s: s.pop(), "exactly 2 sections"),
    (lambda s: s[0]["questions"].pop(), "exactly 4 questions"),
    (lambda s: s[0]["questions"][0].update(options=["a", "b", "c"]), "exactly 4 options"),
    (lambda s: s[0]["questions"][0].update(options=["a", "a", "b", "c"]), "different"),
    (lambda s: s[0]["questions"][0].update(options=["a", "b", "c", "All of the above"]), "All of the above"),
    (lambda s: s[0]["questions"][0].update(answer_index=7), "answer_index"),
    (lambda s: s[1]["questions"][0]["subparts"][1].update(marks=9), "add up to 10"),
    (lambda s: s[1]["questions"][0]["subparts"][1].update(marks=None), "give every sub-part"),
    (lambda s: s[1]["questions"][0].update(subparts=[]), "needs 2 sub-parts"),
    (lambda s: s[1]["questions"][0].update(answer=""), "marking points"),
    (lambda s: s[0]["questions"][1].update(text=s[0]["questions"][0]["text"]), "duplicate"),
    (lambda s: s[0]["questions"][0].update(text="   "), "empty"),
])
def test_drafts_that_break_the_structure_are_rejected(mutate, problem):
    sections = draft().model_dump()["sections"]
    mutate(sections)
    with pytest.raises((ep.ExamPaperInvalid, ValueError)) as exc:
        ep.build_paper(draft(sections=sections), PLAN, MATCHES, "cells")
    assert problem in ep.describe_problem(exc.value)


def test_unknown_source_markers_leave_questions_without_a_source():
    sections = draft().model_dump()["sections"]
    sections[0]["questions"][0]["source"] = "S99"
    paper = ep.build_paper(draft(sections=sections), PLAN, MATCHES, "cells")
    assert paper.sections[0].questions[0].source is None
    assert ep.cited_pages(paper) == {("Week1.pdf", 3), ("Week2.pdf", 7)}


def test_custom_structure_keeps_the_students_layout_but_still_checks_questions():
    custom = draft(custom_structure=True, sections=[
        {"title": "Multiple choice", "question_type": "mcq", "questions": [mcq(i) for i in range(10)]}])
    paper = ep.build_paper(custom, PLAN, MATCHES, "cells")
    assert paper.question_count() == 10 and not paper.follows_past_papers
    assert paper.duration_minutes is None and paper.general_instructions == []
    assert paper.total_marks == 10
    bad = draft(custom_structure=True, sections=[
        {"title": "MCQ", "question_type": "mcq", "questions": [mcq(0, options=["only", "two"])]}])
    with pytest.raises(ep.ExamPaperInvalid):
        ep.build_paper(bad, PLAN, MATCHES, "cells")


def test_true_false_questions_get_fixed_options():
    plan = eb.ExamBlueprint.model_validate({"sections": [
        {"question_type": "true_false", "question_count": 2, "marks_per_question": 1}]})
    tf = draft(sections=[{"questions": [
        {"text": "Mitochondria make ATP.", "answer_index": 0},
        {"text": "Ribosomes store DNA.", "answer_index": 1}]}])
    paper = ep.build_paper(tf, plan, MATCHES, "cells")
    assert paper.sections[0].questions[0].options == ["True", "False"]
    assert paper.sections[0].title == "Section A"
    with pytest.raises(ep.ExamPaperInvalid):
        ep.build_paper(draft(sections=[{"questions": [{"text": "x"}, {"text": "y"}]}]), plan,
                       MATCHES, "cells")


def test_unknown_marks_give_no_total():
    plan = eb.ExamBlueprint.model_validate({"sections": [
        {"question_type": "essay", "question_count": 1}]})
    essay = draft(sections=[{"questions": [{"text": "Discuss osmosis.", "answer": "points"}]}])
    paper = ep.build_paper(essay, plan, MATCHES, "cells")
    assert paper.total_marks is None and paper.sections[0].questions[0].marks is None


def test_messages_include_the_structure_and_retry_reason():
    messages = ep.build_messages("/exam cells", "[S1] text", "(none)", PLAN, problem="Section A: x")
    prompt = messages[1]["content"]
    assert "Section A: 4 multiple-choice questions, 1 mark each, 4 options each" in prompt
    assert "{blueprint}" not in prompt and "[S1] text" in prompt
    assert "Section A: x" in messages[2]["content"]


def test_ready_message_never_mentions_answers_being_included():
    paper = ep.build_paper(draft(), PLAN, MATCHES, "cells")
    text = ep.ready_message(60, paper)
    assert "past papers" in text and "no answer key" in text and "1 hour" in text


@pytest.mark.parametrize("mutate, problem", [
    (lambda s: s[0]["questions"][1].update(text=s[0]["questions"][0]["text"].rstrip("?") + " "),
     "duplicate"),
    (lambda s: s[0]["questions"][0].update(options=["Water", "water.", "Salt", "Sugar"]), "different"),
    (lambda s: s[1]["questions"][0]["subparts"][1].update(
        text=s[1]["questions"][0]["subparts"][0]["text"].upper()), "sub-parts must all be different"),
])
def test_near_duplicates_are_rejected(mutate, problem):
    sections = draft().model_dump()["sections"]
    mutate(sections)
    with pytest.raises(ep.ExamPaperInvalid) as exc:
        ep.build_paper(draft(sections=sections), PLAN, MATCHES, "cells")
    assert problem in str(exc.value)


def test_options_differing_only_by_an_inner_symbol_are_allowed():
    sections = draft().model_dump()["sections"]
    sections[0]["questions"][0]["options"] = ["x > y", "x < y", "x = y", "x ≥ y"]
    paper = ep.build_paper(draft(sections=sections), PLAN, MATCHES, "cells")
    assert paper.sections[0].questions[0].options[0] == "x > y"
