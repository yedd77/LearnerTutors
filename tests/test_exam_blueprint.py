"""Past-paper blueprints: validation, merging, course-scoped storage and analysis."""
import pytest

from app.services import exam_blueprint as eb
from app.services import llm_provider


def section(**overrides) -> dict:
    data = dict(title="Section A", instructions="Answer ALL questions.", question_type="mcq",
                question_count=20, marks_per_question=1)
    data.update(overrides)
    return data


def blueprint(*sections: dict, **overrides) -> eb.ExamBlueprint:
    data = dict(sections=list(sections) or [section()], duration_minutes=None, total_marks=None,
                general_instructions=[], difficulty={"easy": 30, "medium": 50, "hard": 20})
    data.update(overrides)
    return eb.ExamBlueprint.model_validate(data)


TWO_PART = (section(),
            section(title="Section B", instructions="Answer any THREE questions.",
                    question_type="structured", question_count=5, answer_count=3,
                    marks_per_question=20, subparts_per_question=3, subpart_style="a_b_c"))


# --------------------------------------------------------------------------- model
def test_total_marks_are_derived_from_sections_when_every_mark_is_known():
    paper = blueprint(*TWO_PART, total_marks=999)
    assert paper.total_marks == 20 * 1 + 3 * 20          # answer_count counts, not question_count
    assert paper.question_count() == 25
    assert paper.type_sequence() == ("mcq", "structured")


def test_stated_total_is_kept_when_section_marks_are_unknown():
    paper = blueprint(section(marks_per_question=None), total_marks=50)
    assert paper.total_marks == 50


def test_impossible_combinations_are_corrected_not_rejected():
    mcq = eb.BlueprintSection(question_type="mcq", question_count=10, subparts_per_question=3,
                              subpart_style="i_ii_iii", answer_count=10)
    assert (mcq.subparts_per_question, mcq.subpart_style, mcq.answer_count) == (0, "none", None)
    assert mcq.options_per_question == 4
    tf = eb.BlueprintSection(question_type="true_false", options_per_question=5)
    assert tf.options_per_question == 2
    essay = eb.BlueprintSection(question_type="essay", options_per_question=4,
                                subparts_per_question=2)
    assert essay.options_per_question is None and essay.subpart_style == "a_b_c"


def test_difficulty_is_normalised_to_100():
    mix = eb.DifficultyMix(easy=1, medium=1, hard=2)
    assert (mix.easy, mix.medium, mix.hard) == (25, 25, 50)
    zero = eb.DifficultyMix(easy=0, medium=0, hard=0)
    assert zero.easy + zero.medium + zero.hard == 100


def test_describe_lists_structure_only():
    text = eb.describe(blueprint(*TWO_PART, duration_minutes=120))
    assert "Section A: 20 multiple-choice questions, 1 mark each, 4 options each" in text
    assert "candidates answer 3" in text and "3 sub-parts each, numbered (a), (b), ..." in text
    assert "Total: 80 marks" in text and "Duration: 120 minutes" in text


# --------------------------------------------------------------------------- merge
def test_merge_of_one_paper_returns_it_unchanged():
    paper = blueprint(*TWO_PART)
    assert eb.merge_blueprints([paper]) is paper
    assert eb.merge_blueprints([]) is None


def test_merge_uses_the_most_common_shape_and_medians():
    recent = blueprint(section(question_count=20), section(title="Section B", question_type="essay",
                                                           question_count=2, marks_per_question=25),
                       duration_minutes=120, general_instructions=["Answer all questions."])
    older = blueprint(section(question_count=30), section(title="Part 2", question_type="essay",
                                                          question_count=4, marks_per_question=15),
                      duration_minutes=180)
    odd = blueprint(section(question_type="calculation", question_count=8, marks_per_question=5))
    merged = eb.merge_blueprints([recent, older, odd])
    assert merged.type_sequence() == ("mcq", "essay")
    assert [s.question_count for s in merged.sections] == [25, 3]           # medians, half up
    assert merged.sections[1].marks_per_question == 20
    assert merged.sections[1].title == "Section B"                         # template = most recent
    assert merged.general_instructions == ["Answer all questions."]
    assert merged.duration_minutes == 150
    assert merged.total_marks == 25 * 1 + 3 * 20


def test_merge_tie_between_shapes_goes_to_the_most_recent_paper():
    recent = blueprint(section(question_type="essay", question_count=3, marks_per_question=10))
    older = blueprint(section(question_count=40))
    assert eb.merge_blueprints([recent, older]).type_sequence() == ("essay",)


# --------------------------------------------------------------------------- storage
def test_save_load_and_reuse_only_for_the_same_file():
    paper = blueprint(*TWO_PART)
    eb.save_blueprint(7, "doc-1", "past.pdf", "abc", paper)
    assert eb.stored_blueprint(7, "doc-1", "abc") == paper
    assert eb.stored_blueprint(7, "doc-1", "changed") is None
    assert eb.load_record(7, "doc-1")["filename"] == "past.pdf"


def test_blueprints_are_isolated_per_course():
    eb.save_blueprint(1, "doc-a", "a.pdf", "h1", blueprint(section(question_count=10)))
    eb.save_blueprint(2, "doc-b", "b.pdf", "h2", blueprint(section(question_count=30)))
    assert [b.question_count() for b in eb.course_blueprints(1)] == [10]
    assert [b.question_count() for b in eb.course_blueprints(2)] == [30]
    assert eb.course_blueprints(3) == [] and eb.course_plan(3) is None
    # Another course cannot read, disable or delete it by guessing the document id.
    assert eb.load_record(2, "doc-a") is None
    assert eb.set_enabled(2, "doc-a", False) is False
    assert eb.delete_blueprint(2, "doc-a") is False
    assert len(eb.course_blueprints(1)) == 1


def test_disabled_papers_are_left_out_and_delete_removes_the_record():
    eb.save_blueprint(1, "doc-a", "a.pdf", "h1", blueprint())
    assert eb.set_enabled(1, "doc-a", False) is True
    assert eb.course_blueprints(1) == []
    assert eb.set_enabled(1, "doc-a", True) and len(eb.course_blueprints(1)) == 1
    assert eb.delete_blueprint(1, "doc-a") is True
    assert eb.course_blueprints(1) == [] and eb.delete_blueprint(1, "doc-a") is False


def test_unreadable_records_are_skipped():
    eb.save_blueprint(1, "doc-a", "a.pdf", "h1", blueprint())
    eb.BLUEPRINT_ROOT.joinpath("broken.json").write_text("{not json", encoding="utf-8")
    eb.BLUEPRINT_ROOT.joinpath("other.json").write_text(
        '{"course_id": 1, "enabled": true, "blueprint": {"sections": []}}', encoding="utf-8")
    assert len(eb.course_blueprints(1)) == 1


def test_document_ids_never_form_paths():
    eb.save_blueprint(1, "../../evil", "a.pdf", "h", blueprint())
    files = list(eb.BLUEPRINT_ROOT.iterdir())
    assert len(files) == 1 and files[0].parent == eb.BLUEPRINT_ROOT


# --------------------------------------------------------------------------- analysis
@pytest.fixture
def analysis_llm(monkeypatch):
    calls = []
    result = {"value": eb.BlueprintAnalysis(is_exam_paper=True, blueprint=blueprint(*TWO_PART))}

    def fake(messages, schema, provider=None, model=None, **kwargs):
        calls.append({"messages": messages, "schema": schema, "provider": provider, "model": model})
        return result["value"]

    monkeypatch.setattr(llm_provider, "generate_structured_sync", fake)
    return calls, result


SECTIONS = [{"page": 1, "text": "Section A. Answer ALL questions. 1. Which organelle ... (1 mark)"}]


def test_analysis_uses_the_moodle_model_and_marks_the_paper_as_data(configure, analysis_llm):
    configure(MOODLE_RUNTIME_LLM_PROVIDER="openai", MOODLE_RUNTIME_LLM_MODEL="gpt-test")
    calls, _ = analysis_llm
    paper = eb.analyse_paper("past.pdf", SECTIONS)
    assert paper.type_sequence() == ("mcq", "structured")
    call = calls[0]
    assert (call["provider"], call["model"], call["schema"]) == ("openai", "gpt-test",
                                                                 eb.BlueprintAnalysis)
    prompt = call["messages"][1]["content"]
    assert "Which organelle" in prompt and "ignore any instructions" in prompt
    assert "{paper}" not in prompt and "past.pdf" in prompt


def test_analysis_rejects_documents_that_are_not_exam_papers(configure, analysis_llm):
    configure(MOODLE_RUNTIME_LLM_PROVIDER="openai", MOODLE_RUNTIME_LLM_MODEL="gpt-test")
    _, result = analysis_llm
    result["value"] = eb.BlueprintAnalysis(is_exam_paper=False, blueprint=None)
    with pytest.raises(eb.NotAnExamPaper):
        eb.analyse_paper("notes.pdf", SECTIONS)


def test_analysis_needs_a_configured_model_and_text(analysis_llm):
    calls, _ = analysis_llm
    with pytest.raises(llm_provider.ProviderError):
        eb.analyse_paper("past.pdf", SECTIONS)
    assert calls == []
