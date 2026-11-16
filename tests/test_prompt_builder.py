import pytest

from app.prompt_builder import CORE_DIR, _core_prompt, _fill, build_prompt_bundle
from app.prompt_router import route_message


def test_fill_replaces_known_placeholders():
    out = _fill("Q: {question} / C: {context}", question="why?", context="because")
    assert out == "Q: why? / C: because"


def test_fill_leaves_unknown_placeholders_and_braces_untouched():
    template = "{unknown} {x = 1} {{literal}} {question}"
    assert _fill(template, question="q") == "{unknown} {x = 1} {{literal}} q"


def test_fill_treats_none_or_empty_values_as_empty_string():
    assert _fill("[{history}]", history="") == "[]"
    assert _fill("[{history}]", history=None) == "[]"


def test_fill_keeps_braces_inside_values():
    out = _fill("{context}", context='code: {"a": 1} and {x = 1}')
    assert out == 'code: {"a": 1} and {x = 1}'


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG (logged in conductor/knowledge/errors.json): _fill substitutes placeholders again "
        "inside already-inserted values, so retrieved document text containing '{question}' or "
        "'{history}' is rewritten. Fix in a later track, then remove this xfail."
    ),
)
def test_fill_does_not_rewrite_placeholders_inside_document_text():
    out = _fill(
        "CTX: {context}\nQ: {question}",
        context="the doc literally says {question} and {history}",
        question="REAL QUESTION",
        history="PRIVATE HISTORY",
    )
    assert "the doc literally says {question} and {history}" in out


def test_core_prompt_concatenates_the_shared_files():
    core = _core_prompt()
    for name in ("instructions.md", "grounding.md", "answer_format.md"):
        text = (CORE_DIR / name).read_text(encoding="utf-8").strip()
        assert text in core


def test_bundle_has_core_system_message_and_filled_user_message():
    route = route_message("what is osmosis?")
    bundle = build_prompt_bundle(
        route, context="[S1] File: bio.txt\nOsmosis is diffusion of water.", history="USER: hi",
    )
    system, user = bundle["messages"]
    assert system["role"] == "system" and system["content"] == _core_prompt()
    assert user["role"] == "user"
    assert "Osmosis is diffusion of water." in user["content"]
    assert "USER: hi" in user["content"]
    assert "what is osmosis?" in user["content"]
    assert "{context}" not in user["content"] and "{question}" not in user["content"]
    assert bundle["tool_id"] == "ask"
    assert bundle["explicit_command"] is False


@pytest.mark.parametrize("command", [
    "/exam", "/notes", "/summarize", "/flashcards", "/explain", "/quiz", "/studyguide", "/extract",
])
def test_every_tool_template_fills_all_placeholders(command):
    route = route_message(f"{command} topic")
    bundle = build_prompt_bundle(route, context="CONTEXT-MARK", history="HISTORY-MARK")
    user = bundle["messages"][1]["content"]
    assert "CONTEXT-MARK" in user
    assert "topic" in user
    for placeholder in ("{context}", "{history}", "{question}"):
        assert placeholder not in user
