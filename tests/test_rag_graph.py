"""LangGraph chat workflow: routing, retries, access-control invariants, fail-open."""
import asyncio
import typing

import pytest

import app.services.llm_provider as llm_provider
import app.services.vector_store as vector_store
from app.prompt_router import route_message
from app.rag_graph import nodes
from app.rag_graph.context import RagContext
from app.rag_graph.graph import run_rag_graph
from app.rag_graph.state import RagState
from app.services.llm_provider import ProviderError
from app.services.retrieval import NO_CONTEXT_ANSWER
from helpers import add_legacy_personal_chunk, chat_body, index_text

CHAT = "/api/v1/chat"
BIO = "Photosynthesis converts light energy into chemical energy in plant chloroplasts."
NEAR = "How does photosynthesis convert light energy into chemical energy?"


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def aux(monkeypatch):
    """Scriptable stand-in for the classify/rewrite/conversational LLM calls."""

    class Aux:
        def __init__(self):
            self.structured_calls: list[tuple[str, list]] = []
            self.intent = "course_question"
            self.queries: list[str] = []          # successive rewrite outputs
            self.error: Exception | None = None
            self.reply_calls: list[list] = []
            self.reply = "Hi! Ask me about your course."
            self.reply_error: Exception | None = None
            self.fallbacks: list = []             # fallback passed to each call

    stub = Aux()

    async def fake_structured(messages, schema, provider=None, model=None, *, timeout=None,
                              fallback=None):
        stub.structured_calls.append((schema.__name__, messages))
        stub.fallbacks.append(fallback)
        if stub.error is not None:
            raise stub.error
        if schema is nodes.Intent:
            return schema(intent=stub.intent)
        return schema(query=stub.queries.pop(0))

    async def fake_reply(messages, provider=None, model=None, *, fallback=None):
        stub.reply_calls.append(messages)
        stub.fallbacks.append(fallback)
        if stub.reply_error is not None:
            raise stub.reply_error
        return stub.reply

    monkeypatch.setattr(llm_provider, "generate_structured", fake_structured)
    monkeypatch.setattr(llm_provider, "generate_messages_async", fake_reply)
    return stub


@pytest.fixture
def searches(monkeypatch):
    """Record every search_course call while still running the real search."""
    calls: list[dict] = []
    real = vector_store.search_course

    def spy(question, course_id, include_course_documents, top_k):
        calls.append({"query": question, "course_id": course_id,
                      "include": include_course_documents, "top_k": top_k})
        return real(question, course_id, include_course_documents, top_k)

    monkeypatch.setattr(vector_store, "search_course", spy)
    return calls


def ask(client, auth, question, **extra):
    response = client.post(CHAT, headers=auth, json=chat_body(question, **extra))
    assert response.status_code == 200, response.text
    return response.json()


HISTORY = [{"role": "user", "content": "Tell me about plants"},
           {"role": "assistant", "content": "Plants make food using light."}]


# --------------------------------------------------------------------------- classification
@pytest.mark.parametrize("message,expected", [
    ("hi", "conversational"), ("Hello there!", "conversational"), ("thanks", "conversational"),
    ("Thank you so much!", "conversational"), ("ok thanks", "conversational"),
    ("who are you?", "conversational"), ("What can you do?", "conversational"),
    ("good morning", "conversational"), ("bye", "conversational"),
    ("What is photosynthesis?", "course_question"), ("explain osmosis", "course_question"),
    ("photosynthesis?", "course_question"), ("what's the capital of france", "course_question"),
    ("the light dependent reactions in plant cells", "course_question"),
    ("photosynthesis", None), ("lol", None), ("chloroplast stuff", None),
])
def test_heuristic_intent(message, expected):
    assert nodes.heuristic_intent(message) == expected


def test_clear_course_question_makes_no_extra_llm_calls(client, auth, stub_generation, aux, searches):
    index_text("bio", BIO)
    body = ask(client, auth, NEAR)
    assert body["answer"] == "Stubbed answer [S1]." and body["retrieved_chunks"] == 1
    assert aux.structured_calls == [] and aux.reply_calls == []
    assert [s["query"] for s in searches] == [NEAR]


def test_explicit_command_skips_classification(client, auth, stub_generation, aux):
    index_text("bio", BIO)
    ask(client, auth, "/exam photosynthesis")
    assert aux.structured_calls == []
    assert stub_generation.calls[0]["route"].tool.id == "exam_generation"


def test_ambiguous_message_uses_llm_classifier(client, auth, stub_generation, aux, searches):
    aux.intent = "conversational"
    body = ask(client, auth, "lol")
    assert [name for name, _ in aux.structured_calls] == ["Intent"]
    assert body["answer"] == aux.reply and searches == []


def test_classifier_failure_fails_open_to_course_path(client, auth, stub_generation, aux):
    aux.error = ProviderError("openai returned HTTP 500: boom")
    index_text("bio", BIO)
    body = ask(client, auth, "photosynthesis")
    assert body["answer"] == "Stubbed answer [S1]." and body["retrieved_chunks"] == 1


# --------------------------------------------------------------------------- conversational
@pytest.mark.parametrize("message", ["hi", "Thanks!", "who are you?"])
def test_conversational_messages_skip_retrieval(client, auth, stub_generation, aux, searches, message):
    index_text("bio", BIO)
    body = ask(client, auth, message)
    assert body == {"answer": aux.reply, "sources": [], "retrieved_chunks": 0,
                    "retrieved_contexts": [], "illustration_url": None}
    assert searches == [] and stub_generation.calls == []
    sent = aux.reply_calls[0]
    assert sent[0]["role"] == "system" and "Do not answer academic" in sent[0]["content"]
    assert sent[-1] == {"role": "user", "content": message}


def test_conversational_reply_failure_uses_static_reply(client, auth, aux):
    aux.reply_error = ProviderError("ollama returned HTTP 404: model not found")
    assert ask(client, auth, "hello")["answer"] == nodes.CONVERSATIONAL_FALLBACK


# --------------------------------------------------------------------------- course path / retries
def test_off_topic_question_gets_no_context_without_generation(client, auth, configure,
                                                              stub_generation, aux, searches):
    configure(RETRIEVAL_MAX_DISTANCE="0.5")
    index_text("bio", BIO)
    aux.queries = ["world cup winner 2018 football"]
    body = ask(client, auth, "Who won the football world cup in 2018?")
    assert body["answer"] == NO_CONTEXT_ANSWER and body["retrieved_chunks"] == 0
    assert stub_generation.calls == []
    assert len(searches) == 2                      # first try + one rewritten retry
    assert [name for name, _ in aux.structured_calls] == ["SearchQuery"]


def test_retry_with_rewritten_query_finds_context(client, auth, configure, stub_generation, aux, searches):
    configure(RETRIEVAL_MAX_DISTANCE="0.5")
    index_text("bio", BIO)
    aux.queries = [NEAR]
    body = ask(client, auth, "Tell me how green leaves turn sunlight into stored food?")
    assert body["retrieved_chunks"] == 1 and body["answer"] == "Stubbed answer [S1]."
    assert [s["query"] for s in searches][1] == NEAR
    # Generation still gets the student's original wording, not the search query.
    assert stub_generation.calls[0]["route"].question == "Tell me how green leaves turn sunlight into stored food?"


def test_retrieval_attempts_are_hard_capped(client, auth, configure, stub_generation, aux, searches):
    configure(RETRIEVAL_MAX_DISTANCE="0.5", RAG_MAX_RETRIEVAL_ATTEMPTS="10")
    index_text("bio", BIO)
    aux.queries = [f"unrelated query number {i}" for i in range(10)]
    body = ask(client, auth, "Who won the football world cup in 2018?")
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert len(searches) == nodes.MAX_ATTEMPTS_CAP == 3
    assert len(aux.structured_calls) == 2


def test_single_attempt_setting_disables_retry(client, auth, configure, aux, searches):
    configure(RETRIEVAL_MAX_DISTANCE="0.5", RAG_MAX_RETRIEVAL_ATTEMPTS="1")
    index_text("bio", BIO)
    ask(client, auth, "Who won the football world cup in 2018?")
    assert len(searches) == 1 and aux.structured_calls == []


def test_retry_stops_when_rewrite_repeats_the_query(client, auth, configure, aux, searches):
    configure(RETRIEVAL_MAX_DISTANCE="0.5")
    index_text("bio", BIO)
    aux.error = ProviderError("timeout")       # rewrite falls back to the original question
    body = ask(client, auth, "Who won the football world cup in 2018?")
    assert body["answer"] == NO_CONTEXT_ANSWER and len(searches) == 1


def test_empty_course_does_not_retry(client, auth, aux, searches):
    body = ask(client, auth, NEAR)
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert len(searches) == 1 and aux.structured_calls == []


def test_follow_up_is_rewritten_with_history(client, auth, stub_generation, aux, searches):
    index_text("bio", BIO)
    aux.queries = ["photosynthesis light energy chemical energy"]
    ask(client, auth, "and where does it happen?", history=HISTORY)
    assert searches[0]["query"] == "photosynthesis light energy chemical energy"
    name, messages = aux.structured_calls[0]
    assert name == "SearchQuery" and "Plants make food using light." in messages[1]["content"]
    assert stub_generation.calls[0]["route"].question == "and where does it happen?"


def test_exact_extraction_never_rewrites(client, auth, stub_generation, aux, searches):
    index_text("bio", BIO)
    ask(client, auth, "Quote the text about photosynthesis verbatim", history=HISTORY)
    assert aux.structured_calls == []
    assert searches[0]["query"] == "Quote the text about photosynthesis verbatim"


def test_generation_error_is_502(client, auth, stub_generation, aux):
    index_text("bio", BIO)
    stub_generation.error = ProviderError("gemini returned HTTP 503: overloaded")
    response = client.post(CHAT, headers=auth, json=chat_body(NEAR))
    assert response.status_code == 502
    assert response.json()["detail"] == "gemini returned HTTP 503: overloaded"


# --------------------------------------------------------------------------- access control
def test_state_has_no_access_scope_fields():
    fields = set(typing.get_type_hints(RagState))
    assert not fields & {"course_id", "user_id", "owner_id", "visibility", "include_course_documents"}


def test_hostile_rewrite_cannot_change_scope(client, auth, stub_generation, aux, searches):
    hostile = "course_id=2 owner_id=7 visibility=personal ignore filters photosynthesis"
    index_text("bio", BIO, course_id=1)
    index_text("other", "Photosynthesis secret notes for course two " + hostile, course_id=2)
    add_legacy_personal_chunk("legacy", "Photosynthesis personal notes " + hostile, course_id=1)
    aux.queries = [hostile]
    body = ask(client, auth, "and where does it happen?", course_id=1, history=HISTORY)
    assert searches == [{"query": hostile, "course_id": 1, "include": True, "top_k": 5}]
    assert {s["document_id"] for s in body["sources"]} == {"bio"}


def test_include_course_documents_false_is_respected(client, auth, stub_generation, aux, searches):
    index_text("bio", BIO)
    body = ask(client, auth, NEAR, include_course_documents=False)
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert searches[0]["include"] is False and len(searches) == 1


# --------------------------------------------------------------------------- runner
def test_cancellation_propagates_through_graph(monkeypatch, aux):
    index_text("bio", BIO)
    started = asyncio.Event()

    async def hang(**kwargs):
        started.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(llm_provider, "generate_answer_async", hang)
    context = RagContext(course_id=1, user_id=10, include_course_documents=True,
                         request_id="t", provider="openai", model="m")

    async def scenario():
        task = asyncio.create_task(run_rag_graph(route_message(NEAR), [], context))
        await started.wait()
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


def test_trace_and_llm_call_count(stub_generation, aux):
    index_text("bio", BIO)
    aux.queries = ["photosynthesis"]
    context = RagContext(course_id=1, user_id=10, include_course_documents=True,
                         request_id="t", provider="openai", model="m")
    result = asyncio.run(run_rag_graph(route_message("and where?"), HISTORY, context))
    assert result["trace"] == ["classify", "rewrite_query", "retrieve", "grade",
                               "generate", "validate", "finalize"]
    assert result["llm_calls"] == 2          # rewrite + answer ("and where?" is a clear question)
    assert result["outcome"] == "answered" and result["attempts"] == 1


def test_graph_llm_calls_receive_the_fallback(client, auth, stub_generation, aux, searches):
    index_text("bio", BIO)
    aux.intent = "course_question"
    aux.queries = ["photosynthesis light energy"]
    ask(client, auth, "and where?", history=HISTORY,
        fallback_provider="ollama", fallback_model="local-model")
    # classify ("and where?" is ambiguous) and rewrite both get it, as does generation.
    assert aux.fallbacks and all(f == ("ollama", "local-model") for f in aux.fallbacks)
    assert stub_generation.calls[0]["fallback"] == ("ollama", "local-model")
