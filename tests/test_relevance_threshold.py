"""Relevance threshold: off-topic questions get the no-context reply."""
import logging

import pytest

import app.services.retrieval as retrieval_module
from app.config import CALIBRATED_MAX_DISTANCE, Settings
from helpers import chat_body, index_text

CHAT = "/api/v1/chat"
NO_CONTEXT_PREFIX = "I couldn't find relevant text in the documents available to you"
CHUNK = "Photosynthesis converts light energy into chemical energy in plant chloroplasts."


def ask(client, auth, question):
    return client.post(CHAT, headers=auth, json=chat_body(question))


# --------------------------------------------------------------------------- threshold resolution
def settings(**values) -> Settings:
    values.setdefault("retrieval_max_distance", None)  # conftest pins the env value to 0
    return Settings(_env_file=None, **values)


def test_calibrated_default_for_current_embedding_model():
    s = settings(embedding_provider="openai", embedding_model="text-embedding-3-small")
    assert s.resolved_retrieval_max_distance == 0.70
    assert CALIBRATED_MAX_DISTANCE[("openai", "text-embedding-3-small")] == 0.70


def test_default_model_name_resolution_is_used():
    s = settings(embedding_provider="openai", embedding_model="", openai_embed_model="text-embedding-3-small")
    assert s.resolved_retrieval_max_distance == 0.70


@pytest.mark.parametrize("provider,model", [("ollama", "nomic-embed-text"), ("gemini", "gemini-embedding-001")])
def test_uncalibrated_models_have_no_default(provider, model):
    assert settings(embedding_provider=provider, embedding_model=model).resolved_retrieval_max_distance == 0.0


@pytest.mark.parametrize("value,expected", [(0.55, 0.55), (0, 0.0), (-1, 0.0)])
def test_explicit_value_overrides_calibration(value, expected):
    s = settings(embedding_provider="openai", embedding_model="text-embedding-3-small",
                 retrieval_max_distance=value)
    assert s.resolved_retrieval_max_distance == expected


# --------------------------------------------------------------------------- chat behaviour
def test_far_question_gets_no_context_answer(client, auth, configure, stub_generation):
    configure(RETRIEVAL_MAX_DISTANCE="0.5")
    index_text("bio", CHUNK)
    body = ask(client, auth, "Who won the football world cup in 2018?").json()
    assert body["retrieved_chunks"] == 0
    assert body["sources"] == [] and body["retrieved_contexts"] == []
    assert body["answer"].startswith(NO_CONTEXT_PREFIX)
    assert stub_generation.calls == []


def test_near_question_is_still_answered(client, auth, configure, stub_generation):
    configure(RETRIEVAL_MAX_DISTANCE="0.5")
    index_text("bio", CHUNK)
    body = ask(client, auth, "How does photosynthesis convert light energy into chemical energy?").json()
    assert body["retrieved_chunks"] == 1
    assert body["answer"] == "Stubbed answer [S1]."
    assert "distance" not in body and all("distance" not in s for s in body["sources"])


def test_only_distant_chunks_are_dropped(client, auth, configure, stub_generation):
    configure(RETRIEVAL_MAX_DISTANCE="0.5")
    index_text("bio", CHUNK, page=1)
    index_text("maths", "Polynomial long division and the remainder theorem.", page=2)
    body = ask(client, auth, "How does photosynthesis convert light energy into chemical energy?").json()
    assert body["retrieved_chunks"] == 1
    assert [s["document_id"] for s in body["sources"]] == ["bio"]


def test_disabled_threshold_keeps_current_behaviour(client, auth, configure, stub_generation):
    configure(RETRIEVAL_MAX_DISTANCE="0")
    index_text("bio", CHUNK)
    body = ask(client, auth, "Who won the football world cup in 2018?").json()
    assert body["retrieved_chunks"] == 1


def test_filtering_is_logged_with_distances(client, auth, configure, stub_generation, caplog):
    configure(RETRIEVAL_MAX_DISTANCE="0.5")
    index_text("bio", CHUNK)
    with caplog.at_level(logging.INFO, logger="app.api.chat"):
        ask(client, auth, "Who won the football world cup in 2018?")
    [line] = [r.getMessage() for r in caplog.records if "chat.retrieval.filtered" in r.getMessage()]
    assert "dropped=1 kept=0 threshold=0.500 best_distance=0." in line


def test_uncalibrated_model_warns_once(client, auth, configure, stub_generation, caplog, monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MAX_DISTANCE")
    monkeypatch.setattr("app.config.Settings.model_config", {**Settings.model_config, "env_file": None})
    configure()
    monkeypatch.setattr(retrieval_module, "_warned_uncalibrated", set())
    index_text("bio", CHUNK)
    with caplog.at_level(logging.WARNING, logger="app.api.chat"):
        ask(client, auth, "Who won the football world cup in 2018?")
        ask(client, auth, "Who won the football world cup in 2018?")
    warnings = [r for r in caplog.records if "threshold_disabled" in r.getMessage()]
    assert len(warnings) == 1
    assert "embedding=openai/test-embed" in warnings[0].getMessage()
