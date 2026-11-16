"""Pick the prompt tool for a chat message (explicit tool, slash command or wording)."""
from dataclasses import dataclass
import re
from typing import Optional
from app.prompt_registry import TOOLS, ToolDefinition, get_tool


# Route a message to the correct tool based on its content
@dataclass(frozen=True)
class RouteResult:
    """The chosen tool, the message text to use (slash command removed), and how it was chosen."""
    tool: ToolDefinition
    question: str
    explicit_command: bool


# Natural-language patterns, checked in order against the lower-cased message.
# The first tool with a matching pattern wins; nothing matching means "ask".
_PATTERN_ROUTES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exact_text_extraction", (
        r"\bverbatim\b",
        r"\bexact wording\b",
        r"\bquote (?:this|the following|the text)\b",
        r"\bextract (?:the exact|verbatim)\b",
        r"\bcopy .+ exactly\b",
    )),
    ("exam_generation", (
        r"\b(?:create|generate|make|write|prepare|design)\b.{0,70}"
        r"\b(?:exam questions?|examination questions?|mock exam|exam paper|"
        r"mcqs?|multiple.choice questions?|test questions?)\b",
    )),
    ("note_making", (r"\b(?:make|create|prepare|write)\b.{0,60}\b(?:revision notes|study notes|notes)\b",)),
    ("summarization", (r"\b(?:summari[sz]e|summarise|summary)\b",)),
    ("flashcards", (r"\bflashcards?\b|\bstudy cards\b",)),
    ("practice_quiz", (r"\b(?:practice quiz|quiz me|test me|practice questions)\b",)),
    ("study_guide", (r"\b(?:study guide|revision plan|study plan)\b",)),
    ("concept_explanation", (r"\b(?:explain|teach me|walk me through)\b",)),
)


def route_message(message: str, requested_tool_id: Optional[str] = None) -> RouteResult:
    """Choose a tool for ``message``, in priority order.

    1. ``requested_tool_id`` from Moodle (unknown IDs raise ValueError); the
       tool's slash command is stripped if the text also starts with it.
    2. A leading slash command such as ``/exam`` (case-insensitive), which is removed.
    3. The first matching pattern in ``_PATTERN_ROUTES``.
    4. The default ``ask`` tool.

    Routing is local and rule-based; it never calls an LLM.
    """
    text = (message or "").strip()

    # Use the explicitly requested tool if provided
    if requested_tool_id:
        tool = get_tool(requested_tool_id)
        # Avoid accidentally dropping text unless this is the selected tool's command.
        if tool.command and text.lower().startswith(tool.command.lower()):
            text = text[len(tool.command):].lstrip()
        return RouteResult(tool=tool, question=text, explicit_command=True)

    # recheck for explicit command in the message text
    parts = text.split(maxsplit=1)
    first = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""
    for tool in TOOLS.values():
        if tool.command and first.lower() == tool.command.lower():
            return RouteResult(tool=tool, question=rest.strip(), explicit_command=True)

    # Check for patterns that indicate a specific tool should be used
    lower = text.lower()
    for tool_id, patterns in _PATTERN_ROUTES:
        if any(re.search(pattern, lower) for pattern in patterns):
            return RouteResult(get_tool(tool_id), text, False)

    return RouteResult(get_tool("ask"), text, False)
