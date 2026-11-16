"""Suggestion parsing and validation (no LLM calls: generation is stubbed)."""
import json

import pytest

import app.services.suggestion_generator as sg
from app.services.llm_provider import ProviderError

FENCE = "`" * 3
SECTIONS = [{"page": 1, "text": "Photosynthesis converts light energy into chemical energy."}]


# --------------------------------------------------------------------------- bulk parsing
@pytest.mark.parametrize("raw", [
    '{"suggestions": ["What is A?", "Why B?", "How C?"]}',
    f'{FENCE}json\n{{"suggestions": ["What is A?", "Why B?", "How C?"]}}\n{FENCE}',
    f'{FENCE}JSON {{"suggestions": ["What is A?", "Why B?", "How C?"]}}{FENCE}',
    'Sure! Here you go: {"suggestions": ["What is A?", "Why B?", "How C?"]} Hope it helps.',
    '["What is A?", "Why B?", "How C?"]',
])
def test_parse_suggestions_accepts_common_shapes(raw):
    assert sg._parse_suggestions(raw) == ["What is A?", "Why B?", "How C?"]


def test_parse_suggestions_cleans_dedupes_and_truncates():
    raw = json.dumps({"suggestions": [
        "  What   is A? ", "what is a", 7, "", "x" * 241, "Why B?", "How C?", "Extra D?",
    ]})
    assert sg._parse_suggestions(raw) == ["What is A?", "Why B?", "How C?"]


@pytest.mark.parametrize("raw,message", [
    ("no json here", "valid JSON object"),
    ('{"other": []}', "suggestions array"),
    ('{"suggestions": ["Only one?", "only one"]}', "Expected 3 distinct suggestions; got 1"),
])
def test_parse_suggestions_rejects_bad_output(raw, message):
    with pytest.raises(ValueError, match=message):
        sg._parse_suggestions(raw)


def test_document_excerpt_is_capped():
    sections = [{"text": "a" * 6000}, {"text": "   "}, {"text": "b" * 6000}, {"text": "c" * 10}]
    excerpt = sg._document_excerpt(sections)
    assert len(excerpt.replace("\n", "")) == sg.MAX_SOURCE_CHARS
    assert "c" not in excerpt


def test_fallback_uses_section_topics_then_generic_questions():
    sections = [{"text": "Photosynthesis converts light. More text."}, {"text": "short"}]
    result = sg._fallback_suggestions("notes/Biology Week 1.pdf", sections)
    assert result == [
        "What are the key ideas and concepts about Photosynthesis converts light?",
        "What are the main concepts explained in Biology Week 1?",
        "What important facts and examples should I understand from Biology Week 1?",
    ]


# --------------------------------------------------------------------------- single suggestion
@pytest.fixture
def single(monkeypatch, configure, runtime_path):
    configure(OPENAI_API_KEY="sk-test-0000000000000000")
    runtime_path.write_text(json.dumps({"llm_provider": "openai", "llm_model": "m"}), encoding="utf-8")
    state = {"raw": '{"suggestion": "What is chlorophyll?"}', "calls": []}

    def fake_generate_messages(messages, provider=None, model=None, json_mode=False):
        state["calls"].append({"messages": messages, "provider": provider, "model": model, "json_mode": json_mode})
        return state["raw"]

    monkeypatch.setattr(sg, "generate_messages", fake_generate_messages)
    return state


@pytest.mark.parametrize("raw", [
    '{"suggestion": "  What is   chlorophyll? "}',
    f'{FENCE}json\n{{"suggestion": "What is chlorophyll?"}}\n{FENCE}',
    f'{FENCE}{{"suggestion": "What is chlorophyll?"}}{FENCE}',
])
def test_single_suggestion_parses(single, raw):
    single["raw"] = raw
    assert sg.generate_one_document_suggestion("bio.pdf", SECTIONS, ["Why plants?"]) == "What is chlorophyll?"
    [call] = single["calls"]
    assert call["json_mode"] is True and (call["provider"], call["model"]) == ("openai", "m")
    assert '["Why plants?"]' in call["messages"][1]["content"]


@pytest.mark.parametrize("raw,existing,message", [
    ("not json", [], "valid JSON object"),
    ('{"other": "x"}', [], "suggestion string"),
    ('{"suggestion": "   "}', [], "empty or too long"),
    ('{"suggestion": "' + "x" * 241 + '"}', [], "empty or too long"),
    ('{"suggestion": "What is chlorophyll"}', ["what is CHLOROPHYLL?"], "duplicates"),
])
def test_single_suggestion_rejects_bad_output(single, raw, existing, message):
    single["raw"] = raw
    with pytest.raises(ValueError, match=message):
        sg.generate_one_document_suggestion("bio.pdf", SECTIONS, existing)


def test_single_suggestion_requires_model_and_text(single, runtime_path):
    with pytest.raises(ValueError, match="No extractable document text"):
        sg.generate_one_document_suggestion("bio.pdf", [{"text": "  "}])
    runtime_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProviderError, match="No generation provider/model"):
        sg.generate_one_document_suggestion("bio.pdf", SECTIONS)
