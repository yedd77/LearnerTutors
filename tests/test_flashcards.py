"""/flashcards structured revision decks: requested count, the content model,
the graph node (sufficiency, retry, failure) and the response contract."""
import asyncio

import pytest
from pydantic import ValidationError

import app.services.llm_provider as llm_provider
from app.prompt_router import route_message
from app.rag_graph.context import RagContext
from app.rag_graph.graph import run_rag_graph
from app.rag_graph.nodes import Ranking
from app.services import flashcards as decks
from app.services.llm_provider import ProviderError
from helpers import chat_body, index_text

CHAT = "/api/v1/chat"
TOPIC = "photosynthesis light energy chloroplast glucose"
LECTURE = "Week 2 - Photosynthesis.pdf"


def material(n: int, course_id: int = 1, words: int = 60) -> None:
    """Index ``n`` on-topic chunks (separate documents) of about ``words`` words each."""
    for i in range(n):
        body = " ".join([f"Photosynthesis fact {i}:"] + ["light energy chloroplast glucose"] * (words // 4))
        index_text(f"c{course_id}card{i}", body, course_id=course_id,
                   filename=LECTURE if course_id == 1 else f"Course{course_id}_{i}.pdf", page=i + 1)


def card(n: int, source: str = "S1", **overrides) -> dict:
    data = dict(front=f"What is fact {n} about photosynthesis?",
                back=f"Fact {n} explains how light energy is stored.", source=source)
    data.update(overrides)
    return data


def sample_draft(count: int = 5, **overrides) -> decks.FlashcardDraft:
    data = dict(title="Photosynthesis", coverage="sufficient",
                cards=[card(i) for i in range(count)])
    data.update(overrides)
    return decks.FlashcardDraft(**data)


@pytest.fixture
def card_llm(monkeypatch):
    """Stub the structured flashcards call; ``drafts`` are returned in turn (last one repeats)."""

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
        assert schema is decks.FlashcardDraft, f"unexpected structured call: {schema.__name__}"
        stub.calls.append({"messages": messages, "provider": provider, "model": model})
        if stub.error is not None:
            raise stub.error
        return stub.drafts[min(len(stub.calls), len(stub.drafts)) - 1]

    monkeypatch.setattr(llm_provider, "generate_structured", fake_structured)
    return stub


def chat(client, auth, text=f"/flashcards {TOPIC}", **extra):
    response = client.post(CHAT, headers=auth, json=chat_body(text, **extra))
    assert response.status_code == 200, response.text
    return response.json()


def run_graph(text: str = f"/flashcards {TOPIC}") -> dict:
    """Run the RAG graph directly, to inspect outcome and llm_calls."""
    context = RagContext(course_id=1, user_id=10, include_course_documents=True,
                         request_id="test", provider="openai", model="test-model")
    return asyncio.run(run_rag_graph(route=route_message(text), history=[], context=context))


# --------------------------------------------------------------------------- requested count
@pytest.mark.parametrize("text, expected", [
    (TOPIC, 5),
    ("10 cards on mitosis", 10),
    ("20 flashcards on mitosis", 15),
    ("2 cards on mitosis", 3),
    ("week 3 cell structure", 5),
    ("chapter 2, 8 flash cards", 8),
    ("12 Flashcards", 12),
    ("give me 1 card", 3),
])
def test_requested_count(text, expected):
    assert decks.requested_count(text) == expected


# --------------------------------------------------------------------------- content model
MATCHES = [{"text": "t", "metadata": {"filename": "Lecture 6.pdf", "page": 14}},
           {"text": "t", "metadata": {"filename": "Lecture 7.pdf", "page": 3}}]


def test_valid_draft_builds_a_deck_and_tidies_whitespace():
    draft = sample_draft(title="  Photo   synthesis ")
    draft.cards[0].front = "  What is   fact 0? "
    deck = decks.build_deck(draft, MATCHES, "fallback", 5)
    assert deck.title == "Photo synthesis" and len(deck.cards) == 5
    assert deck.cards[0].front == "What is fact 0?"
    assert deck.cards[0].source == decks.FlashcardSource(filename="Lecture 6.pdf", page=14)


def test_markers_map_to_passages_the_model_saw():
    cards = [card(0, source="S2"), card(1, source="S99"), card(2, source="Lecture 6.pdf"),
             card(3, source="")]
    deck = decks.build_deck(sample_draft(cards=cards), MATCHES, "fallback", 5)
    assert deck.cards[0].source == decks.FlashcardSource(filename="Lecture 7.pdf", page=3)
    assert all(c.source is None for c in deck.cards[1:])


def test_blank_title_falls_back_to_the_topic():
    deck = decks.build_deck(sample_draft(title="  "), [], "x" * 200, 5)
    assert deck.title == "x" * 160


@pytest.mark.parametrize("draft, requested, message", [
    (sample_draft(6), 5, "between 3 and 5 cards"),        # more than requested: not trimmed
    (sample_draft(2), 5, "between 3 and 5 cards"),
    (sample_draft(16), 15, "between 3 and 15 cards"),
    (sample_draft(cards=[card(0), card(0, front="WHAT IS FACT 0 ABOUT PHOTOSYNTHESIS?"), card(1)]),
     5, "duplicate card fronts"),
    (sample_draft(cards=[card(0, back="what is fact 0 about photosynthesis?"), card(1), card(2)]),
     5, "not repeat it"),
    (sample_draft(cards=[card(0, back=""), card(1), card(2)]), 5, "at least 1 character"),
    (sample_draft(cards=[card(0, front="f" * 301), card(1), card(2)]), 5, "at most 300"),
    (sample_draft(cards=[card(0, back="b" * 801), card(1), card(2)]), 5, "at most 800"),
])
def test_invalid_decks_are_rejected(draft, requested, message):
    with pytest.raises(ValidationError, match=message):
        decks.build_deck(draft, [], "fallback", requested)


def test_deck_without_context_is_capped_at_max_cards():
    with pytest.raises(ValidationError):
        decks.FlashcardDeck(title="T", cards=[{"front": f"F{i}", "back": "B"} for i in range(16)])


# --------------------------------------------------------------------------- graph node
def test_too_little_material_is_insufficient_without_an_llm_call(card_llm):
    index_text("tiny", "Photosynthesis uses light.")
    result = run_graph()
    assert result["outcome"] == "insufficient" and result["flashcards"] is None
    assert "couldn't find enough material" in result["answer"]
    assert card_llm.calls == [] and result["llm_calls"] == 0


def test_invalid_then_valid_draft_is_retried_once(card_llm):
    material(5)
    card_llm.drafts = [sample_draft(7), sample_draft(5)]   # 7 > the default 5
    result = run_graph()
    assert result["outcome"] == "flashcards" and len(result["flashcards"].cards) == 5
    assert result["llm_calls"] == 2
    retry = card_llm.calls[1]["messages"]
    assert len(retry) == 3 and "Your previous flashcards could not be used" in retry[2]["content"]
    assert "between 3 and 5 cards" in retry[2]["content"]


def test_two_invalid_drafts_fail(card_llm):
    material(5)
    card_llm.drafts = [sample_draft(2)]
    result = run_graph()
    assert result["outcome"] == "flashcards_failed" and result["flashcards"] is None
    assert result["answer"] == decks.failed_message() and result["llm_calls"] == 2


def test_limited_draft_with_two_cards_is_insufficient(card_llm):
    material(5)
    card_llm.drafts = [sample_draft(2, coverage="limited")]
    result = run_graph()
    assert result["outcome"] == "insufficient" and result["llm_calls"] == 1
    assert result["answer"] == decks.limited_message(TOPIC)


def test_limited_draft_with_four_cards_is_a_four_card_deck(card_llm):
    material(5)
    card_llm.drafts = [sample_draft(4, coverage="limited")]
    result = run_graph(f"/flashcards 10 cards on {TOPIC}")
    assert result["outcome"] == "flashcards" and len(result["flashcards"].cards) == 4
    assert result["answer"].startswith("Here are 4 flashcards on Photosynthesis.")


def test_prompt_carries_the_count_and_this_course_only(card_llm):
    material(5)
    index_text("other", "Secret course two photosynthesis glucose", course_id=2)
    run_graph(f"/flashcards 8 cards on {TOPIC}")
    [call] = card_llm.calls
    system, user = call["messages"][0]["content"], call["messages"][1]["content"]
    assert system.startswith("You write accurate revision flashcards") and "Do not invent facts" in system
    assert "## Number of cards requested\n8" in user and "Write 8 cards." in user
    assert LECTURE in user and "[S1]" in user and "Secret course two" not in user
    assert "{count}" not in user and "{context}" not in user and "/flashcards" not in user


# --------------------------------------------------------------------------- endpoint
def test_flashcards_reply_carries_the_deck(client, auth, card_llm):
    material(5)
    body = chat(client, auth)
    deck = body["flashcards"]
    assert deck["title"] == "Photosynthesis" and len(deck["cards"]) == 5
    assert set(deck["cards"][0]) == {"front", "back", "source"}
    assert deck["cards"][0]["source"] == {"filename": LECTURE, "page": 1}
    assert body["answer"] == "Here are 5 flashcards on Photosynthesis. Tap a card to see the answer."
    assert "fact" not in body["answer"].lower()          # no card text in the history line
    assert "quiz" not in body and "pending_action" not in body and "download" not in body
    assert body["illustration_url"] is None
    assert [s["filename"] for s in body["sources"]] == [LECTURE]   # only the cited page (S1)
    assert body["retrieved_chunks"] == 5 and len(body["retrieved_contexts"]) == 5


def test_thin_material_is_plain_text_with_no_pending_action(client, auth, card_llm):
    index_text("tiny", "Photosynthesis uses light.")
    body = chat(client, auth)
    assert "couldn't find enough material" in body["answer"]
    assert "flashcards" not in body and "pending_action" not in body and body["sources"] == []


def test_failed_deck_is_plain_text(client, auth, card_llm):
    material(5)
    card_llm.drafts = [sample_draft(2)]
    body = chat(client, auth)
    assert body["answer"] == decks.failed_message() and "flashcards" not in body


def test_provider_failure_while_writing_the_deck_is_a_502(client, auth, card_llm):
    material(5)
    card_llm.error = ProviderError("openai is unavailable", retryable=True)
    response = client.post(CHAT, headers=auth, json=chat_body(f"/flashcards {TOPIC}"))
    assert response.status_code == 502


def test_other_replies_have_no_flashcards_key(client, auth, stub_generation, card_llm):
    material(5)
    assert "flashcards" not in chat(client, auth, f"/summarize {TOPIC}")
    assert "flashcards" not in chat(client, auth, f"What is {TOPIC}?")
    assert card_llm.calls == []
