"""/notes study sheets: the content model, the sufficiency check and LLM generation.

The LLM only produces a ``StudySheet`` (structured content); ``app.pdf``
renders it. Nothing here decides access: the retrieved chunks passed in were
already scoped to the request's course by the graph.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.prompt_builder import CORE_DIR, _fill, _read_prompt
from app.prompt_registry import PROMPT_DIR
from app.services import llm_provider

NOTES_TOOL_ID = "note_making"
EXPAND_ACTION = "expand_notes_search"
PROMPT_FILE = PROMPT_DIR / "graph" / "study_sheet.md"

# Validation limits are deliberately looser than the prompt's length budget:
# content is never cut silently. If a sheet is too long for one page the PDF
# simply gets a second page and the student is told (see app.pdf.study_sheet).
_TEXT = 600
_ITEMS = 12


def _clean(value: str) -> str:
    """Collapse whitespace."""
    return " ".join(str(value).split())


class Definition(BaseModel):
    """A term and its meaning, as given in the course material."""
    term: str = Field(min_length=1, max_length=160)
    meaning: str = Field(min_length=1, max_length=_TEXT)


class Formula(BaseModel):
    """A formula in plain text and what it means."""
    expression: str = Field(min_length=1, max_length=300)
    meaning: str = Field(default="", max_length=_TEXT)


class StudySheet(BaseModel):
    """Content of one study sheet; every list may be empty."""
    title: str = Field(min_length=1, max_length=160)
    coverage: Literal["sufficient", "limited"]
    overview: str = Field(default="", max_length=1200)
    key_concepts: list[str] = Field(default_factory=list, max_length=_ITEMS)
    definitions: list[Definition] = Field(default_factory=list, max_length=_ITEMS)
    main_points: list[str] = Field(default_factory=list, max_length=_ITEMS)
    examples: list[str] = Field(default_factory=list, max_length=_ITEMS)
    formulas: list[Formula] = Field(default_factory=list, max_length=_ITEMS)
    remember: list[str] = Field(default_factory=list, max_length=_ITEMS)
    summary: list[str] = Field(default_factory=list, max_length=_ITEMS)

    @field_validator("title", "overview")
    @classmethod
    def _tidy_text(cls, value: str) -> str:
        """Collapse whitespace in single-text fields."""
        return _clean(value)

    @field_validator("key_concepts", "main_points", "examples", "remember", "summary")
    @classmethod
    def _tidy_items(cls, items: list[str]) -> list[str]:
        """Drop blank bullets and reject over-long ones."""
        cleaned = [_clean(item) for item in items if _clean(item)]
        if any(len(item) > _TEXT for item in cleaned):
            raise ValueError(f"Bullet longer than {_TEXT} characters.")
        return cleaned

    def has_content(self) -> bool:
        """True when there is anything to put on the page besides the title."""
        return bool(self.overview or self.key_concepts or self.definitions or self.main_points
                    or self.examples or self.formulas or self.remember or self.summary)


def is_notes(tool_id: str) -> bool:
    """True for the /notes tool."""
    return tool_id == NOTES_TOOL_ID


def enough_material(matches: list[dict], settings: Settings | None = None) -> bool:
    """Deterministic minimum before any LLM call: enough relevant chunks and text."""
    settings = settings or get_settings()
    chars = sum(len(match.get("text", "")) for match in matches)
    return (len(matches) >= max(1, settings.notes_min_chunks)
            and chars >= max(0, settings.notes_min_context_chars))


def expanded_top_k(settings: Settings | None = None) -> int:
    """top_k for a widened search (the vector store caps it at 20)."""
    return max(1, min(20, (settings or get_settings()).notes_expanded_top_k))


def build_messages(question: str, context: str, history: str) -> list[dict[str, str]]:
    """Messages for the structured study-sheet call: grounding rules plus the task prompt."""
    grounding = CORE_DIR / "grounding.md"
    rules = _read_prompt(str(grounding)) if grounding.is_file() else ""
    system = ("You write accurate, concise revision content for university students. "
              "Return only valid JSON.\n\n" + rules).strip()
    prompt = _fill(_read_prompt(str(PROMPT_FILE)), context=context, history=history,
                   question=question)
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


async def generate_study_sheet(question: str, context: str, history: list[dict],
                               provider: str, model: str,
                               fallback: tuple[str, str] | None = None) -> StudySheet:
    """Ask the configured model for the sheet content. Raises ProviderError on failure."""
    messages = build_messages(question, context, llm_provider._history_text(history))
    return await llm_provider.generate_structured(messages, StudySheet, provider, model,
                                                  fallback=fallback)


# --------------------------------------------------------------------------- student messages
def _topic(question: str) -> str:
    """The request as a short topic phrase for messages."""
    topic = _clean(question).rstrip("?.! ")
    return topic if len(topic) <= 80 else topic[:77].rstrip() + "..."


def ask_to_expand_message(question: str) -> str:
    """Permission request shown when the first search found too little."""
    return (
        f"I found only limited information about \"{_topic(question)}\" in the course "
        "materials I searched, so a study sheet made from it would be incomplete.\n\n"
        "Would you like me to search more widely across this course's documents to "
        "create more complete notes?"
    )


def still_limited_message(question: str) -> str:
    """Honest reply when even the widened search found too little."""
    return (
        f"I searched more widely across this course's documents, but there still isn't "
        f"enough material about \"{_topic(question)}\" to make a reliable study sheet, so "
        "I haven't created one. Your lecturer may not have uploaded material on this topic "
        "yet. You could try a more specific or differently worded topic."
    )


def declined_message() -> str:
    """Reply when the student declines the widened search."""
    return ("No problem, I've cancelled that study sheet and nothing was generated. "
            "You can send /notes again with a different topic at any time.")


def expired_request_message() -> str:
    """Reply when an approval arrives for an expired or invalid pending request."""
    return ("That request has expired or is no longer valid, so I didn't run it. "
            "Please send your /notes request again.")


def ready_message(ttl_minutes: int, pages: int) -> str:
    """Success message shown with the Download PDF button."""
    if ttl_minutes % 60 == 0 and ttl_minutes >= 60:
        hours = ttl_minutes // 60
        period = f"{hours} hour" + ("s" if hours != 1 else "")
    else:
        period = f"{ttl_minutes} minute" + ("s" if ttl_minutes != 1 else "")
    pages_note = ""
    if pages > 1:
        pages_note = (f"\n\nThis topic needed {pages} pages to stay readable, so the sheet "
                      "runs onto a second page rather than using tiny text.")
    return (
        "Your study sheet is ready!\n\n"
        "Click the button below to download your PDF."
        f"{pages_note}\n\n"
        "Please download and save it to your device. Generated study sheets are only "
        f"available for {period} and are then deleted automatically. They are not stored "
        "permanently in LearnerTutors."
    )


def render_failed_message() -> str:
    """Reply when the PDF could not be produced."""
    return ("Sorry, I couldn't create the study sheet PDF this time. "
            "Please try your /notes request again.")
