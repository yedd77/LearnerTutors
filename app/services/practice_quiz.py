"""/quiz practice quizzes: the content model, the sufficiency check and LLM generation.

The LLM writes a loose ``QuizDraft``; ``build_quiz`` turns it into a validated
``PracticeQuiz`` (exactly 10 questions, 4 distinct options each), mapping each
question's passage marker to that passage's file and page so a source can
never be invented. Moodle renders the quiz as an interactive card. Nothing
here decides access: the chunks passed in were already scoped to the
request's course by the graph.
"""
import random
import re
from collections import Counter
from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import Settings, get_settings
from app.prompt_builder import CORE_DIR, _fill, _read_prompt
from app.prompt_registry import PROMPT_DIR
from app.services import llm_provider

QUIZ_TOOL_ID = "practice_quiz"
EXPAND_ACTION = "expand_quiz_scope"
PROMPT_FILE = PROMPT_DIR / "graph" / "practice_quiz.md"
QUESTION_COUNT = 10
OPTION_COUNT = 4
_TOPIC_CHARS = 120
# Catch-all options that make a question trivial or ambiguous.
_CATCH_ALL = re.compile(r"^(?:all|none|both|neither)\s+of\s+(?:the\s+)?(?:above|these|them)\b"
                        r"|^both\s+[a-d]\s+and\s+[a-d]\b", re.IGNORECASE)
_MARKER = re.compile(r"S(\d+)", re.IGNORECASE)


def _clean(value: str) -> str:
    """Collapse whitespace."""
    return " ".join(str(value).split())


def _tidy(value):
    """Collapse whitespace in strings; leave other values for type validation."""
    return _clean(value) if isinstance(value, str) else value


# --------------------------------------------------------------------------- content model
class QuizSource(BaseModel):
    """The course document page a question comes from."""
    filename: str = Field(min_length=1, max_length=255)
    page: int | None = None


class QuizQuestion(BaseModel):
    """One multiple-choice question with exactly one correct option."""
    question: str = Field(min_length=1, max_length=600)
    options: list[str]
    answer_index: int = Field(ge=0, le=OPTION_COUNT - 1)
    explanation: str = Field(max_length=800)   # 1-2 sentences, why the answer is right
    source: QuizSource | None = None

    @field_validator("question", "explanation", mode="before")
    @classmethod
    def _tidy_text(cls, value):
        """Collapse whitespace."""
        return _tidy(value)

    @field_validator("options", mode="before")
    @classmethod
    def _tidy_options(cls, value):
        """Collapse whitespace in every option."""
        return [_tidy(item) for item in value] if isinstance(value, list) else value

    @field_validator("options")
    @classmethod
    def _check_options(cls, options: list[str]) -> list[str]:
        """Exactly 4 distinct options of 1-300 characters, none a catch-all."""
        if len(options) != OPTION_COUNT:
            raise ValueError(f"Each question needs exactly {OPTION_COUNT} options.")
        if any(not 1 <= len(option) <= 300 for option in options):
            raise ValueError("Each option must be 1-300 characters.")
        if len({option.casefold() for option in options}) != OPTION_COUNT:
            raise ValueError("The options of a question must all be different.")
        if any(_CATCH_ALL.match(option) for option in options):
            raise ValueError('Options such as "All of the above" are not allowed.')
        return options


class PracticeQuiz(BaseModel):
    """Exactly 10 different questions on one (possibly broadened) topic."""
    title: str = Field(min_length=1, max_length=160)
    questions: list[QuizQuestion]

    @field_validator("title", mode="before")
    @classmethod
    def _tidy_title(cls, value):
        """Collapse whitespace."""
        return _tidy(value)

    @field_validator("questions")
    @classmethod
    def _check_questions(cls, questions: list[QuizQuestion]) -> list[QuizQuestion]:
        """Exactly 10 questions and no duplicates."""
        if len(questions) != QUESTION_COUNT:
            raise ValueError(f"The quiz needs exactly {QUESTION_COUNT} questions.")
        if len({q.question.casefold() for q in questions}) != QUESTION_COUNT:
            raise ValueError("The quiz contains duplicate questions.")
        return questions


# --------------------------------------------------------------------------- LLM output
class QuizDraftQuestion(BaseModel):
    """A question as the model writes it; checked by ``build_quiz``."""
    question: str = ""
    options: list[str] = Field(default_factory=list)
    answer_index: int = 0
    explanation: str = ""
    source: str = ""          # passage marker from the context, e.g. "S3"


class QuizDraft(BaseModel):
    """Structured output of the quiz call. Deliberately loose, so a bad quiz is
    caught by ``build_quiz`` (and retried) instead of failing as a provider error."""
    title: str = ""
    coverage: Literal["sufficient", "limited"] = "sufficient"
    supported_questions: int = QUESTION_COUNT
    broader_topic: str = ""
    questions: list[QuizDraftQuestion] = Field(default_factory=list)

    def is_limited(self) -> bool:
        """True when the model says the material cannot support a full quiz."""
        return self.coverage == "limited" or self.supported_questions < QUESTION_COUNT


def is_quiz(tool_id: str) -> bool:
    """True for the /quiz tool."""
    return tool_id == QUIZ_TOOL_ID


def _passage_sources(matches: list[dict]) -> dict[int, QuizSource]:
    """Passage number (the n in [Sn]) → its file and page, as numbered in the context."""
    sources = {}
    for number, match in enumerate(matches, 1):
        metadata = match.get("metadata", {})
        filename = _clean(metadata.get("filename", ""))[:255]
        page = metadata.get("page", -1)
        if filename:
            sources[number] = QuizSource(filename=filename,
                                         page=page if isinstance(page, int) and page > 0 else None)
    return sources


def _spread_answers(quiz: PracticeQuiz) -> PracticeQuiz:
    """Move correct answers to varied positions when the model bunched them up.

    10 questions over 4 positions should use each position 2-3 times; if one
    position is used more than 4 times, every correct option is swapped into a
    deterministic, shuffled position. Swapping keeps the option texts intact.
    """
    if max(Counter(q.answer_index for q in quiz.questions).values()) <= OPTION_COUNT:
        return quiz
    positions = (list(range(OPTION_COUNT)) * 3)[:len(quiz.questions)]
    random.Random(quiz.title).shuffle(positions)
    questions = []
    for question, target in zip(quiz.questions, positions):
        options = list(question.options)
        options[question.answer_index], options[target] = options[target], options[question.answer_index]
        questions.append(question.model_copy(update={"options": options, "answer_index": target}))
    return quiz.model_copy(update={"questions": questions})


def build_quiz(draft: QuizDraft, matches: list[dict], fallback_title: str) -> PracticeQuiz:
    """Validate the draft into a ``PracticeQuiz``. Raises ``ValidationError``.

    A source is taken only from a marker that names a passage the model was
    shown; anything else leaves the question without a source.
    """
    sources = _passage_sources(matches)

    def source(marker: str) -> QuizSource | None:
        found = _MARKER.search(marker or "")
        return sources.get(int(found.group(1))) if found else None

    quiz = PracticeQuiz(
        title=_clean(draft.title) or _clean(fallback_title)[:160],
        questions=[{"question": q.question, "options": q.options, "answer_index": q.answer_index,
                    "explanation": q.explanation, "source": source(q.source)}
                   for q in draft.questions],
    )
    return _spread_answers(quiz)


def cited_pages(quiz: PracticeQuiz) -> set[tuple[str, int | None]]:
    """(filename, page) of every question's source."""
    return {(q.source.filename, q.source.page) for q in quiz.questions if q.source}


def describe_problem(exc: ValidationError) -> str:
    """Short, content-free summary of why a draft was rejected (for the retry prompt and logs)."""
    parts = []
    for error in exc.errors()[:6]:
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(parts)[:600]


# --------------------------------------------------------------------------- sufficiency
def _material(matches: list[dict]) -> tuple[int, int]:
    """(chunks, characters) retrieved; merged passages count every chunk they contain."""
    chunks = sum(int(match.get("chunk_count", 1)) for match in matches)
    return chunks, sum(len(match.get("text", "")) for match in matches)


def enough_material(matches: list[dict], settings: Settings | None = None) -> bool:
    """Deterministic minimum before any LLM call: enough relevant chunks and text."""
    settings = settings or get_settings()
    chunks, chars = _material(matches)
    return (chunks >= max(1, settings.quiz_min_chunks)
            and chars >= max(0, settings.quiz_min_context_chars))


def estimated_questions(matches: list[dict], settings: Settings | None = None) -> int:
    """Rough number of questions thin material could support (always below 10)."""
    settings = settings or get_settings()
    per_question = max(1, settings.quiz_min_context_chars // QUESTION_COUNT)
    return min(QUESTION_COUNT - 1, _material(matches)[1] // per_question)


def clean_topic(value: str) -> str:
    """A topic phrase safe to show and to search with: one line, at most 120 characters."""
    topic = _clean(value).strip(" \"'").rstrip(".?!:;, ")
    return topic if len(topic) <= _TOPIC_CHARS else topic[:_TOPIC_CHARS - 3].rstrip() + "..."


def broader_topic(matches: list[dict]) -> str:
    """The section or document the retrieved chunks belong to: the most common
    heading, else the most common file name without its extension."""
    headings = Counter(_clean(m.get("metadata", {}).get("heading") or "") for m in matches)
    headings.pop("", None)
    if headings:
        return clean_topic(headings.most_common(1)[0][0])
    files = Counter(PurePath(str(m.get("metadata", {}).get("filename") or "")).stem
                    for m in matches)
    files.pop("", None)
    return clean_topic(files.most_common(1)[0][0].replace("_", " ")) if files else ""


# --------------------------------------------------------------------------- generation
def build_messages(question: str, context: str, history: str,
                   problem: str = "") -> list[dict[str, str]]:
    """Messages for the structured quiz call: grounding rules plus the task prompt.

    ``problem`` explains why the previous attempt was rejected (retry only).
    """
    grounding = CORE_DIR / "grounding.md"
    rules = _read_prompt(str(grounding)) if grounding.is_file() else ""
    system = ("You write accurate multiple-choice practice questions for university students. "
              "Return only valid JSON.\n\n" + rules).strip()
    prompt = _fill(_read_prompt(str(PROMPT_FILE)), context=context, history=history,
                   question=question)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if problem:
        messages.append({"role": "user", "content": (
            f"Your previous quiz could not be used: {problem}\n"
            "Write the whole quiz again and follow every rule: exactly 10 different questions, "
            "each with exactly 4 different options and one correct answer_index (0-3).")})
    return messages


async def generate_draft(question: str, context: str, history: list[dict],
                         provider: str, model: str,
                         fallback: tuple[str, str] | None = None,
                         problem: str = "") -> QuizDraft:
    """Ask the configured model for the quiz. Raises ProviderError on failure."""
    messages = build_messages(question, context, llm_provider._history_text(history), problem)
    return await llm_provider.generate_structured(messages, QuizDraft, provider, model,
                                                  fallback=fallback)


# --------------------------------------------------------------------------- student messages
def _topic(question: str) -> str:
    """The request as a short topic phrase for messages."""
    topic = _clean(question).rstrip("?.! ")
    return topic if len(topic) <= 80 else topic[:77].rstrip() + "..."


def ask_to_expand_message(question: str, supported: int, broader: str) -> str:
    """Yes/no question shown when the material is too thin for 10 questions."""
    enough = (f"enough for about {supported} questions, not {QUESTION_COUNT}" if supported >= 2
              else f"not enough for a {QUESTION_COUNT}-question quiz")
    offer = (f"broaden the topic to \"{broader}\"" if broader
             else "broaden the topic and search more widely across this course's documents")
    return (f"I found only limited material about \"{_topic(question)}\" in the course "
            f"documents — {enough}.\n\n"
            f"Would you like me to {offer} and make a full {QUESTION_COUNT}-question quiz?")


def still_limited_message(question: str) -> str:
    """Honest reply when even the broadened search found too little."""
    return (f"I searched more widely across this course's documents, but there still isn't "
            f"enough material about \"{_topic(question)}\" for a reliable {QUESTION_COUNT}-question "
            "quiz, so I haven't created one. Your lecturer may not have uploaded material on this "
            "topic yet.")


def declined_message(broader: str) -> str:
    """Reply when the student declines broadening the topic."""
    example = f"\"{broader}\"" if broader else "a whole chapter or lecture"
    return f"Okay, I won't create a quiz. Try /quiz with a wider topic, for example {example}."


def expired_request_message() -> str:
    """Reply when an approval arrives for an expired or invalid quiz request."""
    return ("That request has expired or is no longer valid, so I didn't run it. "
            "Please send your /quiz request again.")


def ready_message(title: str) -> str:
    """Short line shown above the quiz card. Never contains questions or answers."""
    return (f"Here's a {QUESTION_COUNT}-question practice quiz on {title}. "
            "Answer each question, then check your score.")


def failed_message() -> str:
    """Reply when no valid quiz came back after the retry."""
    return (f"Sorry, I couldn't create a reliable {QUESTION_COUNT}-question quiz from the course "
            "material this time, so I haven't shown one. Please try /quiz again, perhaps with a "
            "slightly different topic.")
