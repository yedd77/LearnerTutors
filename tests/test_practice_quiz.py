"""/quiz structured practice quizzes: the content model, sufficiency, the
broaden-topic approve/decline flow and the response contract."""
import time

import pytest
from pydantic import ValidationError

import app.services.llm_provider as llm_provider
import app.services.vector_store as vector_store
from app.rag_graph.nodes import Ranking
from app.services import pending_actions
from app.services import practice_quiz as quizzes
from app.services import study_sheet as sheets
from app.services.llm_provider import ProviderError
from helpers import chat_body, index_text

CHAT = "/api/v1/chat"
TOPIC = "photosynthesis light energy chloroplast glucose"
LECTURE = "Photosynthesis light energy.pdf"   # stem is the broader topic offered


def material(n: int, course_id: int = 1, words: int = 60, filename: str = LECTURE) -> None:
    """Index ``n`` on-topic chunks (separate documents) of about ``words`` words each."""
    for i in range(n):
        body = " ".join([f"Photosynthesis fact {i}:"] + ["light energy chloroplast glucose"] * (words // 4))
        index_text(f"c{course_id}quiz{i}", body, course_id=course_id,
                   filename=filename if course_id == 1 else f"Course{course_id}_{i}.pdf", page=i + 1)


def question(n: int, answer_index: int = 1, source: str = "S1", **overrides) -> dict:
    data = dict(question=f"Question {n} about photosynthesis?",
                options=[f"Option {n}a", f"Option {n}b", f"Option {n}c", f"Option {n}d"],
                answer_index=answer_index, explanation=f"Because of fact {n}.", source=source)
    data.update(overrides)
    return data


def sample_draft(**overrides) -> quizzes.QuizDraft:
    data = dict(title="Photosynthesis basics", coverage="sufficient", supported_questions=10,
                broader_topic="", questions=[question(i, answer_index=i % 4) for i in range(10)])
    data.update(overrides)
    return quizzes.QuizDraft(**data)


def valid_quiz_data(**overrides) -> dict:
    data = dict(title="Photosynthesis",
                questions=[{k: v for k, v in question(i).items() if k != "source"} for i in range(10)])
    data.update(overrides)
    return data


@pytest.fixture
def quiz_llm(monkeypatch):
    """Stub the structured quiz call; ``drafts`` are returned in turn (last one repeats)."""

    class Stub:
        def __init__(self):
            self.calls: list = []
            self.drafts = [sample_draft()]
            self.error: Exception | None = None

    stub = Stub()

    async def fake_structured(messages, schema, provider=None, model=None, *, timeout=None,
                              fallback=None):
        if schema is Ranking:          # rerank: keep the fused order
            return Ranking()
        assert schema is quizzes.QuizDraft, f"unexpected structured call: {schema.__name__}"
        stub.calls.append({"messages": messages, "provider": provider, "model": model})
        if stub.error is not None:
            raise stub.error
        return stub.drafts[min(len(stub.calls), len(stub.drafts)) - 1]

    monkeypatch.setattr(llm_provider, "generate_structured", fake_structured)
    return stub


@pytest.fixture
def searches(monkeypatch):
    """Record hybrid_search queries while running the real search."""
    calls: list[dict] = []
    real = vector_store.hybrid_search

    def spy(queries, course_id, include_course_documents, top_k):
        for query in queries:
            calls.append({"query": query, "course_id": course_id, "top_k": top_k})
        return real(queries, course_id, include_course_documents, top_k)

    monkeypatch.setattr(vector_store, "hybrid_search", spy)
    return calls


def chat(client, auth, text=f"/quiz {TOPIC}", **extra):
    response = client.post(CHAT, headers=auth, json=chat_body(text, **extra))
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- content model
def test_valid_quiz_passes_and_tidies_whitespace():
    data = valid_quiz_data(title="  Photo   synthesis ")
    data["questions"][0]["options"][0] = "  Option   0a "
    quiz = quizzes.PracticeQuiz(**data)
    assert quiz.title == "Photo synthesis" and quiz.questions[0].options[0] == "Option 0a"
    assert len(quiz.questions) == 10 and quiz.questions[0].source is None


@pytest.mark.parametrize("count", [0, 9, 11])
def test_quiz_needs_exactly_ten_questions(count):
    data = valid_quiz_data()
    data["questions"] = [{k: v for k, v in question(i).items() if k != "source"} for i in range(count)]
    with pytest.raises(ValidationError, match="exactly 10 questions"):
        quizzes.PracticeQuiz(**data)


def test_duplicate_questions_are_rejected():
    data = valid_quiz_data()
    data["questions"][1]["question"] = data["questions"][0]["question"].upper()
    with pytest.raises(ValidationError, match="duplicate questions"):
        quizzes.PracticeQuiz(**data)


@pytest.mark.parametrize("options, message", [
    (["a", "b", "c"], "exactly 4 options"),
    (["a", "b", "c", "d", "e"], "exactly 4 options"),
    (["Same", "same", "c", "d"], "all be different"),
    (["a", "", "c", "d"], "1-300 characters"),
    (["a", "x" * 301, "c", "d"], "1-300 characters"),
    (["a", "b", "c", "All of the above"], "All of the above"),
    (["a", "b", "c", "None of these"], "All of the above"),
])
def test_options_are_validated(options, message):
    with pytest.raises(ValidationError, match=message):
        quizzes.QuizQuestion(question="Q?", options=options, answer_index=0, explanation="E.")


@pytest.mark.parametrize("index", [-1, 4])
def test_answer_index_must_be_0_to_3(index):
    with pytest.raises(ValidationError):
        quizzes.QuizQuestion(question="Q?", options=["a", "b", "c", "d"], answer_index=index,
                             explanation="E.")


def test_over_long_text_is_rejected():
    with pytest.raises(ValidationError):
        quizzes.QuizQuestion(question="q" * 601, options=["a", "b", "c", "d"], answer_index=0,
                             explanation="E.")
    with pytest.raises(ValidationError):
        quizzes.QuizQuestion(question="Q?", options=["a", "b", "c", "d"], answer_index=0,
                             explanation="e" * 801)


def test_bunched_answers_are_spread_and_keep_the_correct_option():
    draft = sample_draft(questions=[question(i, answer_index=0) for i in range(10)])
    quiz = quizzes.build_quiz(draft, [], "fallback")
    positions = [q.answer_index for q in quiz.questions]
    assert max(positions.count(p) for p in range(4)) <= 3 and len(set(positions)) == 4
    for i, q in enumerate(quiz.questions):
        assert q.options[q.answer_index] == f"Option {i}a"         # still the right answer
        assert sorted(q.options) == sorted(question(i)["options"])   # nothing lost


def test_varied_answers_are_left_alone():
    quiz = quizzes.build_quiz(sample_draft(), [], "fallback")
    assert [q.answer_index for q in quiz.questions] == [i % 4 for i in range(10)]


def test_sources_come_only_from_passages_the_model_saw():
    matches = [{"text": "t", "metadata": {"filename": "Lecture 6.pdf", "page": 14}},
               {"text": "t", "metadata": {"filename": "Lecture 7.pdf", "page": -1}}]
    questions = [question(0, source="S1"), question(1, source="[S2]"),
                 question(2, source="S9"), question(3, source="Lecture 99.pdf")]
    questions += [question(i) for i in range(4, 10)]
    quiz = quizzes.build_quiz(sample_draft(questions=questions), matches, "fallback")
    assert quiz.questions[0].source == quizzes.QuizSource(filename="Lecture 6.pdf", page=14)
    assert quiz.questions[1].source == quizzes.QuizSource(filename="Lecture 7.pdf", page=None)
    assert quiz.questions[2].source is None and quiz.questions[3].source is None


def test_broader_topic_prefers_the_section_heading():
    matches = [{"metadata": {"heading": "Chapter 3: Photosynthesis", "filename": "a.pdf"}},
               {"metadata": {"heading": "Chapter 3: Photosynthesis", "filename": "b.pdf"}},
               {"metadata": {"heading": "", "filename": "b.pdf"}}]
    assert quizzes.broader_topic(matches) == "Chapter 3: Photosynthesis"
    assert quizzes.broader_topic([{"metadata": {"filename": "Week_4_Plants.pdf"}}]) == "Week 4 Plants"
    assert quizzes.broader_topic([]) == ""


# --------------------------------------------------------------------------- endpoint: quiz
def test_enough_material_returns_a_quiz(client, auth, quiz_llm):
    material(5)
    body = chat(client, auth)
    quiz = body["quiz"]
    assert quiz["title"] == "Photosynthesis basics" and len(quiz["questions"]) == 10
    first = quiz["questions"][0]
    assert set(first) == {"question", "options", "answer_index", "explanation", "source"}
    assert len(first["options"]) == 4 and 0 <= first["answer_index"] <= 3
    assert first["source"]["filename"] == LECTURE and first["source"]["page"]
    assert body["answer"].startswith("Here's a 10-question practice quiz on Photosynthesis basics.")
    # Neither questions nor answers leak into the plain answer Moodle sends back as history.
    assert "Question 0" not in body["answer"] and "Option" not in body["answer"]
    assert "pending_action" not in body and "download" not in body
    assert [s["filename"] for s in body["sources"]] == [LECTURE]   # only the cited page (S1)
    assert body["retrieved_chunks"] == 5 and len(body["retrieved_contexts"]) == 5


def test_quiz_prompt_is_grounded_in_this_course_only(client, auth, quiz_llm):
    material(5)
    index_text("other", "Secret course two photosynthesis glucose", course_id=2)
    chat(client, auth)
    [call] = quiz_llm.calls
    system, user = call["messages"][0]["content"], call["messages"][1]["content"]
    assert "Do not invent facts" in system
    assert LECTURE in user and "[S1]" in user and "Secret course two" not in user
    assert TOPIC in user and "/quiz" not in user
    assert call["provider"] == "openai" and call["model"] == "test-model"


def test_tool_id_selects_the_quiz_without_a_slash_command(client, auth, quiz_llm):
    material(5)
    body = chat(client, auth, TOPIC, tool_id="practice_quiz")
    assert len(body["quiz"]["questions"]) == 10


def test_other_replies_have_no_quiz_key(client, auth, stub_generation, quiz_llm):
    material(5)
    body = chat(client, auth, f"/summarize {TOPIC}")
    assert body["answer"] == "Stubbed answer [S1]." and "quiz" not in body
    body = chat(client, auth, f"What is {TOPIC}?")
    assert "quiz" not in body and quiz_llm.calls == []


def test_invalid_quiz_is_retried_once_with_the_reason(client, auth, quiz_llm):
    material(5)
    bad = sample_draft(questions=[question(i) for i in range(9)])
    quiz_llm.drafts = [bad, sample_draft()]
    body = chat(client, auth)
    assert len(body["quiz"]["questions"]) == 10 and len(quiz_llm.calls) == 2
    retry = quiz_llm.calls[1]["messages"]
    assert len(retry) == 3 and "could not be used" in retry[2]["content"]
    assert "exactly 10 questions" in retry[2]["content"]


def test_two_invalid_quizzes_give_an_honest_plain_text_reply(client, auth, quiz_llm):
    material(5)
    duplicate_options = [question(i, options=["x", "x", "y", "z"]) for i in range(10)]
    quiz_llm.drafts = [sample_draft(questions=duplicate_options)]
    body = chat(client, auth)
    assert "couldn't create a reliable 10-question quiz" in body["answer"]
    assert "quiz" not in body and "pending_action" not in body and body["sources"] == []
    assert len(quiz_llm.calls) == 2


def test_provider_failure_while_writing_the_quiz_is_a_502(client, auth, quiz_llm):
    material(5)
    quiz_llm.error = ProviderError("openai is unavailable", retryable=True)
    response = client.post(CHAT, headers=auth, json=chat_body(f"/quiz {TOPIC}"))
    assert response.status_code == 502


# --------------------------------------------------------------------------- endpoint: too thin
def test_too_little_material_offers_a_broader_topic_without_calling_the_llm(client, auth, quiz_llm):
    material(2)
    body = chat(client, auth)
    assert "quiz" not in body and quiz_llm.calls == []
    action = body["pending_action"]
    assert action["type"] == "expand_quiz_scope"
    assert action["approve_label"] == "Yes, broaden the topic"
    assert action["decline_label"] == "No, don't create it"
    assert f'limited material about "{TOPIC}"' in body["answer"]
    assert "enough for about" in body["answer"] and "not 10" in body["answer"]
    assert 'broaden the topic to "Photosynthesis light energy"' in body["answer"]
    assert body["sources"] == [] and body["retrieved_contexts"] == []
    request = pending_actions.verify(action["token"], quizzes.EXPAND_ACTION, course_id=1, user_id=10)
    assert request.tool_id == "practice_quiz" and request.question == TOPIC
    assert request.topic == "Photosynthesis light energy"


def test_model_judging_material_limited_also_offers_a_broader_topic(client, auth, quiz_llm):
    material(5)
    quiz_llm.drafts = [sample_draft(coverage="limited", supported_questions=4,
                                    broader_topic="Chapter 3: Photosynthesis", questions=[])]
    body = chat(client, auth)
    assert body["pending_action"]["type"] == "expand_quiz_scope" and "quiz" not in body
    assert "enough for about 4 questions" in body["answer"]
    assert 'broaden the topic to "Chapter 3: Photosynthesis"' in body["answer"]


def test_approval_searches_the_broader_topic_in_the_same_course(
        client, auth, quiz_llm, searches, configure):
    configure(QUIZ_MIN_CHUNKS="11", QUIZ_MIN_CONTEXT_CHARS="0")
    material(12)
    material(3, course_id=2)
    first = chat(client, auth)
    assert first["pending_action"]["type"] == "expand_quiz_scope" and searches[-1]["top_k"] == 10
    assert quiz_llm.calls == []

    second = chat(client, auth, "Yes, broaden the topic",
                  pending_action_token=first["pending_action"]["token"], pending_decision="approve")
    assert len(second["quiz"]["questions"]) == 10 and "pending_action" not in second
    assert searches[-1] == {"query": "Photosynthesis light energy", "course_id": 1, "top_k": 20}
    assert second["retrieved_chunks"] == 12          # course 2 was never searched
    assert "## Student topic\nPhotosynthesis light energy" in quiz_llm.calls[0]["messages"][1]["content"]


def test_broadened_search_still_too_thin_is_reported_honestly(client, auth, quiz_llm):
    material(2)
    first = chat(client, auth)
    body = chat(client, auth, "yes", pending_action_token=first["pending_action"]["token"],
                pending_decision="approve")
    assert "still isn't enough material" in body["answer"]
    assert "quiz" not in body and "pending_action" not in body   # no endless re-asking
    assert quiz_llm.calls == []


def test_decline_creates_nothing(client, auth, quiz_llm, searches):
    material(2)
    first = chat(client, auth)
    count = len(searches)
    body = chat(client, auth, "No", pending_action_token=first["pending_action"]["token"],
                pending_decision="decline")
    assert body["answer"] == ("Okay, I won't create a quiz. Try /quiz with a wider topic, "
                              'for example "Photosynthesis light energy".')
    assert "quiz" not in body and "pending_action" not in body
    assert len(searches) == count and quiz_llm.calls == []


@pytest.mark.parametrize("action, tool_id", [
    (quizzes.EXPAND_ACTION, "note_making"),     # quiz action smuggled onto another tool
    (quizzes.EXPAND_ACTION, "ask"),
    (sheets.EXPAND_ACTION, "practice_quiz"),    # notes action replayed for the quiz
])
def test_token_is_rejected_for_another_tool(client, auth, quiz_llm, searches, action, tool_id):
    material(5)
    token, _ = pending_actions.issue(action, course_id=1, user_id=10, question=TOPIC,
                                     tool_id=tool_id, topic="Photosynthesis")
    body = chat(client, auth, "Yes", pending_action_token=token, pending_decision="approve")
    assert "expired or is no longer valid" in body["answer"]
    assert searches == [] and quiz_llm.calls == [] and "quiz" not in body


@pytest.mark.parametrize("change", ["other_user", "other_course", "tampered_topic"])
def test_quiz_token_is_bound_to_its_user_course_and_topic(client, auth, quiz_llm, searches, change):
    token, _ = pending_actions.issue(quizzes.EXPAND_ACTION, course_id=1, user_id=10,
                                     question=TOPIC, tool_id="practice_quiz", topic="Photosynthesis")
    extra = {}
    if change == "other_user":
        extra["user_id"] = 11
    elif change == "other_course":
        extra["course_id"] = 2
    else:
        body, sig = token.rsplit(".", 1)
        forged, _ = pending_actions.issue(quizzes.EXPAND_ACTION, course_id=1, user_id=10,
                                          question=TOPIC, tool_id="practice_quiz",
                                          topic="Everything in course 2")
        token = forged.rsplit(".", 1)[0] + "." + sig
    body = chat(client, auth, "Yes", pending_action_token=token, pending_decision="approve", **extra)
    assert "expired or is no longer valid" in body["answer"]
    assert searches == [] and quiz_llm.calls == []


def test_expired_quiz_token_asks_for_the_quiz_again(client, auth, quiz_llm, searches, monkeypatch):
    token, _ = pending_actions.issue(quizzes.EXPAND_ACTION, course_id=1, user_id=10,
                                     question=TOPIC, tool_id="practice_quiz", topic="Photosynthesis")
    later = time.time() + 31 * 60  # default TTL is 30 minutes
    monkeypatch.setattr(pending_actions.time, "time", lambda: later)
    body = chat(client, auth, "Yes", pending_action_token=token, pending_decision="approve")
    assert "expired or is no longer valid" in body["answer"] and "/quiz" in body["answer"]
    assert searches == [] and quiz_llm.calls == []


def test_notes_tokens_carry_no_topic_field():
    token, _ = pending_actions.issue(sheets.EXPAND_ACTION, course_id=1, user_id=10,
                                     question=TOPIC, tool_id="note_making")
    import base64
    import json
    body = token.rsplit(".", 1)[0]
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    assert "b" not in payload
    assert pending_actions.verify(token, sheets.EXPAND_ACTION, course_id=1, user_id=10).topic == ""
