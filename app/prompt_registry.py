from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# Define a tool with its metadata and prompt file
@dataclass(frozen=True)
class ToolDefinition:
    id: str
    label: str
    description: str
    prompt_file: str
    command: Optional[str] = None
    retrieval_mode: str = "course_rag"
    top_k: int | None = None
    aliases: tuple[str, ...] = ()

# Define every supported tool with its metadata and prompt file
TOOLS: dict[str, ToolDefinition] = {
    "ask": ToolDefinition(
        id="ask",
        label="Ask a course question",
        description="Answer a question using relevant course materials.",
        prompt_file="ask.md",
        retrieval_mode="course_rag",
        aliases=("question", "chat"),
    ),
    "exam_generation": ToolDefinition(
        id="exam_generation",
        label="Generate exam questions",
        description="Create original exam questions from course material.",
        prompt_file="exam_generation.md",
        command="/exam",
        retrieval_mode="course_rag",
        top_k=10,
        aliases=("exam", "mock exam", "exam questions", "simple exam question"),
    ),
    "note_making": ToolDefinition(
        id="note_making",
        label="Make revision notes",
        description="Turn course material into structured revision notes.",
        prompt_file="note_making.md",
        command="/notes",
        retrieval_mode="course_rag",
        top_k=10,
        aliases=("notes", "revision notes", "study notes"),
    ),
    "summarization": ToolDefinition(
        id="summarization",
        label="Summarize material",
        description="Summarize relevant course material.",
        prompt_file="summarization.md",
        command="/summarize",
        retrieval_mode="course_rag",
        top_k=10,
        aliases=("summary", "summarize", "summarise"),
    ),
    "flashcards": ToolDefinition(
        id="flashcards",
        label="Generate flashcards",
        description="Create revision question-and-answer cards.",
        prompt_file="flashcards.md",
        command="/flashcards",
        retrieval_mode="course_rag",
        aliases=("flashcard", "cards"),
    ),
    "concept_explanation": ToolDefinition(
        id="concept_explanation",
        label="Explain a concept",
        description="Explain a topic step by step.",
        prompt_file="concept_explanation.md",
        command="/explain",
        retrieval_mode="course_rag",
        aliases=("explain", "teach me"),
    ),
    "practice_quiz": ToolDefinition(
        id="practice_quiz",
        label="Practice quiz",
        description="Generate a practice quiz from course material.",
        prompt_file="practice_quiz.md",
        command="/quiz",
        retrieval_mode="course_rag",
        top_k=10,
        aliases=("quiz", "practice test"),
    ),
    "study_guide": ToolDefinition(
        id="study_guide",
        label="Build a study guide",
        description="Organize topics for revision.",
        prompt_file="study_guide.md",
        command="/studyguide",
        retrieval_mode="course_rag",
        top_k=10,
        aliases=("study guide", "revision plan"),
    ),
    "exact_text_extraction": ToolDefinition(
        id="exact_text_extraction",
        label="Extract exact text",
        description="Extract or quote text from supplied source material.",
        prompt_file="exact_text_extraction.md",
        command="/extract",
        retrieval_mode="course_rag",
        aliases=("exact text", "verbatim", "quote exactly"),
    ),
}

# Return a supported tool; reject unknown IDs
def get_tool(tool_id: str) -> ToolDefinition:
    try:
        return TOOLS[tool_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported tool_id: {tool_id}") from exc

# return the list of supported tools for a UI command menu
def get_tool_menu() -> list[dict[str, str]]:
    return [
        {
            "id": tool.id,
            "command": tool.command or "",
            "label": tool.label,
            "description": tool.description,
        }
        for tool in TOOLS.values()
    ]