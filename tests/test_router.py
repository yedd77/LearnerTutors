import pytest

from app.prompt_registry import TOOLS, get_tool, get_tool_menu
from app.prompt_router import route_message


def test_exam_command():
    result = route_message("/exam Create 10 questions on chapter 2")
    assert result.tool.id == "exam_generation"
    assert result.explicit_command is True
    assert result.question == "Create 10 questions on chapter 2"


@pytest.mark.parametrize("command,tool_id", [
    ("/exam", "exam_generation"),
    ("/notes", "note_making"),
    ("/summarize", "summarization"),
    ("/flashcards", "flashcards"),
    ("/explain", "concept_explanation"),
    ("/quiz", "practice_quiz"),
    ("/studyguide", "study_guide"),
])
def test_every_slash_command_routes_to_its_tool(command, tool_id):
    result = route_message(f"{command} cell division")
    assert result.tool.id == tool_id
    assert result.explicit_command is True
    assert result.question == "cell division"


def test_slash_command_is_case_insensitive():
    assert route_message("/EXAM topic").tool.id == "exam_generation"


def test_command_must_be_the_first_word():
    result = route_message("please run /exam for me")
    assert result.explicit_command is False


def test_command_must_match_whole_word():
    # "/examine" is not "/exam".
    assert route_message("/examine this").tool.id == "ask"


def test_command_without_text_leaves_empty_question():
    result = route_message("/exam")
    assert result.tool.id == "exam_generation"
    assert result.question == ""


@pytest.mark.parametrize("message,tool_id", [
    ("create exam questions on thermodynamics", "exam_generation"),
    ("Generate a mock exam for week 4", "exam_generation"),
    ("make revision notes on the Krebs cycle", "note_making"),
    ("please summarize chapter 3", "summarization"),
    ("give me a summary of the lecture", "summarization"),
    ("flashcards for week 3", "flashcards"),
    ("quiz me on cell biology", "practice_quiz"),
    ("I need a study guide for the midterm", "study_guide"),
    ("explain osmosis to me", "concept_explanation"),
    # Quoting/verbatim wording has no dedicated tool; it falls through to the default.
    ("what is the exact wording of the definition?", "ask"),
    ("quote the following text verbatim", "ask"),
    ("what is osmosis?", "ask"),
])
def test_natural_language_routing(message, tool_id):
    result = route_message(message)
    assert result.tool.id == tool_id
    assert result.explicit_command is False
    assert result.question == message


def test_explicit_tool_id_takes_precedence_over_text():
    result = route_message("create exam questions on cells", requested_tool_id="flashcards")
    assert result.tool.id == "flashcards"
    assert result.explicit_command is True


def test_explicit_tool_id_strips_only_its_own_command():
    stripped = route_message("/flashcards deck one", requested_tool_id="flashcards")
    assert stripped.question == "deck one"
    # A different tool's command text is user content and must not be dropped.
    kept = route_message("/exam deck one", requested_tool_id="flashcards")
    assert kept.question == "/exam deck one"


def test_explicit_tool_id_without_command_keeps_text():
    assert route_message("anything", requested_tool_id="ask").question == "anything"


def test_unknown_tool_id_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported tool_id"):
        route_message("hello", requested_tool_id="does_not_exist")


@pytest.mark.parametrize("message", ["", "   ", None])
def test_blank_message_defaults_to_ask(message):
    result = route_message(message)
    assert result.tool.id == "ask"
    assert result.question == ""


def test_registry_menu_matches_tools():
    menu = get_tool_menu()
    assert [item["id"] for item in menu] == list(TOOLS)
    assert all({"id", "command", "label", "description"} <= set(item) for item in menu)


def test_every_tool_has_a_prompt_file():
    from app.prompt_registry import PROMPT_DIR

    for tool in TOOLS.values():
        assert (PROMPT_DIR / "tools" / tool.prompt_file).is_file(), tool.id
    assert get_tool("ask").id == "ask"
