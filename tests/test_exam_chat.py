"""/exam through the chat endpoint: past-paper format, PDF download, isolation and fallbacks."""
import asyncio

import pymupdf
import pytest

import app.services.llm_provider as llm_provider
from app.prompt_router import route_message
from app.rag_graph.context import RagContext
from app.rag_graph.graph import run_rag_graph
from app.rag_graph.nodes import Ranking
from app.services import exam_blueprint as eb
from app.services import exam_paper as ep
from helpers import chat_body, index_text

CHAT = "/api/v1/chat"
TOPIC = "photosynthesis light energy chloroplast glucose"
PLAN = eb.ExamBlueprint.model_validate({
    "sections": [
        {"title": "Section A", "instructions": "Answer ALL questions.", "question_type": "mcq",
         "question_count": 3, "marks_per_question": 1},
        {"title": "Section B", "question_type": "structured", "question_count": 1,
         "marks_per_question": 10, "subparts_per_question": 2},
    ],
    "duration_minutes": 60,
})


def material(n: int = 6, course_id: int = 1) -> None:
    for i in range(n):
        body = " ".join([f"Photosynthesis fact {i}:"] + ["light energy chloroplast glucose"] * 25)
        index_text(f"c{course_id}doc{i}", body, course_id=course_id, filename=f"Week{i}.pdf",
                   page=i + 1)


def good_draft(**overrides) -> ep.ExamDraft:
    data = {"title": "Practice Exam: Photosynthesis", "coverage": "sufficient", "sections": [
        {"questions": [{"text": f"Which statement about fact {i} is correct?", "marks": 1,
                        "options": [f"Choice {i}-{k}" for k in range(4)], "answer_index": 1,
                        "source": "S1"} for i in range(3)]},
        {"questions": [{"text": "A plant is kept in bright light.", "answer": "Points ...",
                        "subparts": [{"text": "Describe what happens.", "marks": 4},
                                     {"text": "Explain why.", "marks": 6}], "source": "S2"}]},
    ]}
    data.update(overrides)
    return ep.ExamDraft.model_validate(data)


@pytest.fixture
def exam_llm(monkeypatch):
    """Stub the structured exam call; ``drafts`` are returned in turn (last one repeats)."""

    class Stub:
        def __init__(self):
            self.calls: list = []
            self.drafts = [good_draft()]

    stub = Stub()

    async def fake_structured(messages, schema, provider=None, model=None, *, timeout=None,
                              fallback=None):
        if schema is Ranking:
            return Ranking()
        assert schema is ep.ExamDraft, f"unexpected structured call: {schema.__name__}"
        stub.calls.append({"messages": messages, "provider": provider, "model": model})
        return stub.drafts[min(len(stub.calls), len(stub.drafts)) - 1]

    monkeypatch.setattr(llm_provider, "generate_structured", fake_structured)
    return stub


def chat(client, auth, text=f"/exam {TOPIC}", **extra):
    response = client.post(CHAT, headers=auth, json=chat_body(text, **extra))
    assert response.status_code == 200, response.text
    return response.json()


def run_graph(text: str = f"/exam {TOPIC}", course_id: int = 1) -> dict:
    context = RagContext(course_id=course_id, user_id=10, include_course_documents=True,
                         request_id="test", provider="openai", model="test-model")
    return asyncio.run(run_rag_graph(route=route_message(text), history=[], context=context))


def test_exam_with_past_papers_returns_a_pdf_without_answers(client, auth, exam_llm):
    material()
    eb.save_blueprint(1, "paper-1", "past.pdf", "h", PLAN)
    body = chat(client, auth)
    assert body["download"]["content_type"] == "application/pdf"
    assert body["download"]["filename"].endswith("exam_paper.pdf")
    assert "past papers" in body["answer"] and "no answer key" in body["answer"]
    assert body["sources"] and "quiz" not in body and "flashcards" not in body
    prompt = exam_llm.calls[0]["messages"][1]["content"]
    assert "Section A: 3 multiple-choice questions, 1 mark each" in prompt
    assert len(exam_llm.calls) == 1

    response = client.get(f"/api/v1/generated-files/{body['download']['file_id']}"
                          "?course_id=1&user_id=10", headers=auth)
    assert response.status_code == 200
    doc = pymupdf.open(stream=response.content, filetype="pdf")
    text = " ".join(page.get_text() for page in doc)
    assert "Practice Exam: Photosynthesis" in text and "Section A" in text
    assert "Answer ALL questions." in text and "Time allowed: 1 hour" in text
    assert "Which statement about fact 0 is correct?" in text and "(a)" in text
    assert "Points ..." not in text                       # the model's answer is never printed
    assert "End of paper" in text and "not an official exam" in text


def test_downloaded_paper_belongs_to_the_requesting_student_only(client, auth, exam_llm):
    material()
    eb.save_blueprint(1, "paper-1", "past.pdf", "h", PLAN)
    file_id = chat(client, auth)["download"]["file_id"]
    url = f"/api/v1/generated-files/{file_id}"
    assert client.get(f"{url}?course_id=1&user_id=10", headers=auth).status_code == 200
    assert client.get(f"{url}?course_id=1&user_id=11", headers=auth).status_code == 404
    assert client.get(f"{url}?course_id=2&user_id=10", headers=auth).status_code == 404


def test_without_past_papers_exam_stays_a_chat_answer(client, auth, exam_llm, stub_generation):
    material()
    body = chat(client, auth)
    assert "download" not in body and exam_llm.calls == []
    assert body["answer"]


def test_other_courses_past_papers_are_never_used(client, auth, exam_llm, stub_generation):
    material(course_id=1)
    eb.save_blueprint(2, "paper-2", "other.pdf", "h", PLAN)
    body = chat(client, auth)
    assert "download" not in body and exam_llm.calls == []


def test_disabled_past_paper_is_ignored(client, auth, exam_llm, stub_generation):
    material()
    eb.save_blueprint(1, "paper-1", "past.pdf", "h", PLAN, enabled=False)
    assert "download" not in chat(client, auth)


def test_invalid_paper_is_retried_once_with_the_reason(exam_llm):
    material()
    eb.save_blueprint(1, "paper-1", "past.pdf", "h", PLAN)
    broken = good_draft(sections=good_draft().model_dump()["sections"][:1])
    exam_llm.drafts = [broken, good_draft()]
    result = run_graph()
    assert result["outcome"] == "exam_paper" and len(exam_llm.calls) == 2
    assert "exactly 2 sections" in exam_llm.calls[1]["messages"][-1]["content"]


def test_two_invalid_papers_end_politely(client, auth, exam_llm):
    material()
    eb.save_blueprint(1, "paper-1", "past.pdf", "h", PLAN)
    exam_llm.drafts = [good_draft(sections=[])]
    body = chat(client, auth)
    assert "download" not in body and "couldn't write a practice paper" in body["answer"]
    assert len(exam_llm.calls) == 2 and body["sources"] == []


def test_thin_material_is_reported_without_an_llm_call(client, auth, exam_llm):
    material(n=1)
    eb.save_blueprint(1, "paper-1", "past.pdf", "h", PLAN)
    body = chat(client, auth)
    assert "isn't enough course material" in body["answer"] and exam_llm.calls == []
    assert "download" not in body and "pending_action" not in body


def test_model_judging_material_limited_is_reported(client, auth, exam_llm):
    material()
    eb.save_blueprint(1, "paper-1", "past.pdf", "h", PLAN)
    exam_llm.drafts = [good_draft(coverage="limited", sections=[])]
    body = chat(client, auth)
    assert "isn't enough course material" in body["answer"] and "download" not in body


def test_render_failure_returns_a_polite_message(client, auth, exam_llm, monkeypatch):
    import app.pdf.exam_paper as exam_pdf
    material()
    eb.save_blueprint(1, "paper-1", "past.pdf", "h", PLAN)
    monkeypatch.setattr(exam_pdf, "render_exam_paper", lambda *a, **k: 1 / 0)
    body = chat(client, auth)
    assert body["answer"] == ep.render_failed_message() and "download" not in body


def test_several_papers_are_combined(exam_llm):
    material()
    eb.save_blueprint(1, "paper-1", "a.pdf", "h1", PLAN)
    bigger = PLAN.model_copy(update={"sections": [
        PLAN.sections[0].model_copy(update={"question_count": 5}), PLAN.sections[1]]})
    eb.save_blueprint(1, "paper-2", "b.pdf", "h2", bigger)
    exam_llm.drafts = [good_draft(sections=[
        {"questions": [{"text": f"Q{i}?", "options": ["w", "x", "y", "z"], "answer_index": 0}
                       for i in range(4)]},
        good_draft().model_dump()["sections"][1]])]
    result = run_graph()
    assert result["outcome"] == "exam_paper"
    assert len(result["exam_paper"].sections[0].questions) == 4      # median of 3 and 5
