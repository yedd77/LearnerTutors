from dataclasses import dataclass
import re
from typing import Optional
from app.prompt_registry import ToolDefinition, get_tool

# Route a message to the correct tool based on its content
@dataclass(frozen=True)
class RouteResult:
    tool: ToolDefinition
    question: str
    explicit_command: bool

def route_message(message: str, requested_tool_id: Optional[str] = None) -> RouteResult:
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
    from app.prompt_registry import TOOLS
    for tool in TOOLS.values():
        if tool.command and first.lower() == tool.command.lower():
            return RouteResult(tool=tool, question=rest.strip(), explicit_command=True)

    lower = text.lower()

    # Check for patterns that indicate a specific tool should be used
    exact_patterns = (
        r"\bverbatim\b",
        r"\bexact wording\b",
        r"\bquote (?:this|the following|the text)\b",
        r"\bextract (?:the exact|verbatim)\b",
        r"\bcopy .+ exactly\b",
    )
    
    # If any of the exact patterns match, route to the exact text extraction tool
    if any(re.search(pattern, lower) for pattern in exact_patterns):
        return RouteResult(get_tool("exact_text_extraction"), text, False)
    exam_patterns = (
        r"\b(?:create|generate|make|write|prepare|design)\b.{0,70}"
        r"\b(?:exam questions?|examination questions?|mock exam|exam paper|"
        r"mcqs?|multiple.choice questions?|test questions?)\b",
    )
    if any(re.search(pattern, lower) for pattern in exam_patterns):
        return RouteResult(get_tool("exam_generation"), text, False)

    if re.search(r"\b(?:make|create|prepare|write)\b.{0,60}\b(?:revision notes|study notes|notes)\b", lower):
        return RouteResult(get_tool("note_making"), text, False)

    if re.search(r"\b(?:summari[sz]e|summarise|summary)\b", lower):
        return RouteResult(get_tool("summarization"), text, False)

    if re.search(r"\bflashcards?\b|\bstudy cards\b", lower):
        return RouteResult(get_tool("flashcards"), text, False)

    if re.search(r"\b(?:practice quiz|quiz me|test me|practice questions)\b", lower):
        return RouteResult(get_tool("practice_quiz"), text, False)

    if re.search(r"\b(?:study guide|revision plan|study plan)\b", lower):
        return RouteResult(get_tool("study_guide"), text, False)

    if re.search(r"\b(?:explain|teach me|walk me through)\b", lower):
        return RouteResult(get_tool("concept_explanation"), text, False)

    return RouteResult(get_tool("ask"), text, False)
