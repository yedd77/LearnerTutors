"""disabled_tools: Moodle can turn a tool (e.g. /exam) off per request; routing decides it here."""
import pytest

import app.services.llm_provider as llm_provider
from app.prompt_router import route_message
from app.services import exam_blueprint as eb
from helpers import chat_body, index_text

CHAT = "/api/v1/chat"
OFF = ["exam_generation"]
TURNED_OFF = "The /exam command is turned off in this course."


def chat(client, auth, question, **extra):
    return client.post(CHAT, headers=auth, json=chat_body(question, **extra))


# --------------------------------------------------------------------------- router
def test_explicit_tool_id_of_a_disabled_tool_is_blocked():
    route = route_message("/exam 10 MCQs", requested_tool_id="exam_generation", disabled_tools=OFF)
    assert route.blocked and route.tool.id == "exam_generation"


def test_leading_slash_command_of_a_disabled_tool_is_blocked_not_stripped():
    route = route_message("/exam 10 MCQs on chapter 3", disabled_tools=OFF)
    assert route.blocked and route.question == "/exam 10 MCQs on chapter 3"


@pytest.mark.parametrize("text, expected", [
    ("create mock exam questions on enzymes", "ask"),
    ("make revision notes and exam questions", "note_making"),
    ("generate mcqs and summarise chapter 2", "summarization"),
])
def test_pattern_match_to_a_disabled_tool_falls_through(text, expected):
    assert route_message(text).tool.id == "exam_generation"          # today's behaviour
    route = route_message(text, disabled_tools=OFF)
    assert route.tool.id == expected and not route.blocked


def test_other_tools_are_unaffected():
    assert route_message("/notes cells", disabled_tools=OFF).tool.id == "note_making"
    assert not route_message("/quiz cells", disabled_tools=OFF).blocked


# --------------------------------------------------------------------------- endpoint
@pytest.mark.parametrize("question, extra", [
    ("/exam 10 MCQs on chapter 3", {"tool_id": "exam_generation"}),
    ("10 MCQs on chapter 3", {"tool_id": "exam_generation"}),
    ("/exam 10 MCQs on chapter 3", {}),
    ("/EXAM", {}),                       # blocked before the "enter a request" check
])
def test_disabled_exam_gets_a_plain_answer(client, auth, stub_generation, question, extra):
    response = chat(client, auth, question, disabled_tools=OFF, **extra)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == TURNED_OFF and body["sources"] == []
    assert "download" not in body and "pending_action" not in body
    assert stub_generation.calls == []


def test_pattern_match_is_answered_normally_instead(client, auth, stub_generation):
    index_text("d1", "Enzymes are biological catalysts that speed up reactions.")
    response = chat(client, auth, "create mock exam questions on enzymes", disabled_tools=OFF)
    assert response.status_code == 200
    assert response.json()["answer"] != TURNED_OFF
    assert stub_generation.calls and stub_generation.calls[-1]["route"].tool.id == "ask"


def test_course_with_past_papers_makes_no_exam_paper_when_disabled(client, auth, stub_generation,
                                                                    monkeypatch):
    async def no_structured(*args, **kwargs):
        raise AssertionError("no exam paper may be generated")
    monkeypatch.setattr(llm_provider, "generate_structured", no_structured)
    index_text("d1", "Enzymes are biological catalysts that speed up reactions.")
    eb.save_blueprint(1, "paper-1", "past.pdf", "h", eb.ExamBlueprint.model_validate(
        {"sections": [{"question_type": "mcq", "question_count": 5}]}))
    for question in ("/exam enzymes", "create mock exam questions on enzymes"):
        body = chat(client, auth, question, disabled_tools=OFF).json()
        assert "download" not in body


@pytest.mark.parametrize("disabled", [["ask"], ["not_a_tool"], ["x" * 81],
                                      ["exam_generation"] * 21])
def test_invalid_disabled_tools_are_422(client, auth, stub_generation, disabled):
    assert chat(client, auth, "What are enzymes?", disabled_tools=disabled).status_code == 422


def test_omitted_or_empty_disabled_tools_change_nothing(client, auth, stub_generation):
    index_text("d1", "Enzymes are biological catalysts that speed up reactions.")
    for extra in ({}, {"disabled_tools": []}):
        assert chat(client, auth, "/exam enzymes", **extra).json()["answer"] != TURNED_OFF
