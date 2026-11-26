"""/exam practice papers in the course's past-paper format: content model, checks, generation.

Used only when the course has at least one enabled past-paper blueprint (see
``exam_blueprint``); otherwise /exam stays a normal chat answer. The LLM writes
a loose ``ExamDraft`` from the retrieved course material and the merged
blueprint; ``build_paper`` checks it against that structure and turns it into
an ``ExamPaper`` for the PDF. The model also writes a short answer for every
question so it has to work each one out, but answers never leave this module:
``ExamPaper`` has no answer fields (the developer chose a paper without an
answer key). Nothing here decides access: the chunks passed in were already
scoped to the request's course by the graph.
"""
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import Settings, get_settings
from app.prompt_builder import CORE_DIR, _fill, _read_prompt
from app.prompt_registry import PROMPT_DIR
from app.services import llm_provider
from app.services import practice_quiz as quizzes
from app.services.exam_blueprint import (
    BlueprintSection,
    ExamBlueprint,
    QuestionType,
    SubpartStyle,
    describe,
)

EXAM_TOOL_ID = "exam_generation"
PROMPT_FILE = PROMPT_DIR / "graph" / "exam_paper.md"
MAX_SECTIONS = 10
MAX_QUESTIONS = 60
MIN_CUSTOM_OPTIONS, MAX_CUSTOM_OPTIONS = 3, 6
_MARK_TOLERANCE = 0.01


def _clean(value) -> str:
    """Collapse whitespace."""
    return " ".join(str(value).split()) if value is not None else ""


def _same_key(value: str) -> str:
    """Comparison key for duplicate checks: case, spacing and leading/trailing
    punctuation ignored, so "Water." and "water" or two questions differing only by
    "?" count as the same. Inner symbols are kept, so "x > y" and "x < y" differ."""
    return " ".join(value.casefold().split()).strip(" .,;:!?\"'")


class ExamPaperInvalid(ValueError):
    """The draft does not follow the required structure; the message says why (no content)."""


# --------------------------------------------------------------------------- paper (PDF content)
class ExamSubpart(BaseModel):
    """One sub-part, e.g. (a), of a structured question."""
    text: str = Field(min_length=1, max_length=800)
    marks: float | None = Field(default=None, gt=0, le=100)


class ExamQuestion(BaseModel):
    """One question as printed. There is deliberately no answer field."""
    text: str = Field(min_length=1, max_length=1500)
    marks: float | None = Field(default=None, gt=0, le=100)
    options: list[str] = Field(default_factory=list, max_length=MAX_CUSTOM_OPTIONS)
    subparts: list[ExamSubpart] = Field(default_factory=list, max_length=10)
    source: quizzes.QuizSource | None = None


class ExamSection(BaseModel):
    """A section of the paper with its printed instructions."""
    title: str = Field(min_length=1, max_length=120)
    instructions: str = Field(default="", max_length=400)
    question_type: QuestionType
    answer_count: int | None = None
    subpart_style: SubpartStyle = "none"
    questions: list[ExamQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)


class ExamPaper(BaseModel):
    """A complete practice paper, ready to render. No answers or marking scheme."""
    title: str = Field(min_length=1, max_length=160)
    duration_minutes: int | None = None
    total_marks: float | None = None
    general_instructions: list[str] = Field(default_factory=list, max_length=6)
    sections: list[ExamSection] = Field(min_length=1, max_length=MAX_SECTIONS)
    follows_past_papers: bool = True      # False when the student asked for another structure

    def question_count(self) -> int:
        """Total number of questions."""
        return sum(len(section.questions) for section in self.sections)


# --------------------------------------------------------------------------- LLM output
class ExamDraftSubpart(BaseModel):
    """A sub-part as the model writes it."""
    text: str = ""
    marks: float | None = None


class ExamDraftQuestion(BaseModel):
    """A question as the model writes it; checked by ``build_paper``."""
    text: str = ""
    marks: float | None = None
    options: list[str] = Field(default_factory=list)
    answer_index: int | None = None
    answer: str = ""             # model answer / marking points; used for checking, never printed
    subparts: list[ExamDraftSubpart] = Field(default_factory=list)
    source: str = ""             # passage marker, e.g. "S3"


class ExamDraftSection(BaseModel):
    """A section as the model writes it."""
    title: str = ""
    instructions: str = ""
    question_type: str = ""
    questions: list[ExamDraftQuestion] = Field(default_factory=list)


class ExamDraft(BaseModel):
    """Structured output of the exam call. Deliberately loose, so a bad paper is
    caught by ``build_paper`` (and retried) instead of failing as a provider error."""
    title: str = ""
    coverage: Literal["sufficient", "limited"] = "sufficient"
    custom_structure: bool = False   # the student explicitly asked for a different structure
    sections: list[ExamDraftSection] = Field(default_factory=list)

    @field_validator("custom_structure", mode="before")
    @classmethod
    def _bool(cls, value):
        """Treat null as False."""
        return bool(value)


def is_exam(tool_id: str) -> bool:
    """True for the /exam tool."""
    return tool_id == EXAM_TOOL_ID


# --------------------------------------------------------------------------- checks
def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _MARK_TOLERANCE


def _options(question: ExamDraftQuestion, question_type: str, count: int | None,
             where: str) -> list[str]:
    """Validated, cleaned options for the question type (empty for written questions)."""
    if question_type == "true_false":
        if question.answer_index not in (0, 1):
            raise ExamPaperInvalid(f"{where}: true/false questions need answer_index 0 (True) or 1 (False).")
        return ["True", "False"]
    if question_type != "mcq":
        return []
    options = [_clean(option) for option in question.options]
    if count is not None and len(options) != count:
        raise ExamPaperInvalid(f"{where}: needs exactly {count} options.")
    if count is None and not MIN_CUSTOM_OPTIONS <= len(options) <= MAX_CUSTOM_OPTIONS:
        raise ExamPaperInvalid(f"{where}: needs {MIN_CUSTOM_OPTIONS}-{MAX_CUSTOM_OPTIONS} options.")
    if any(not 1 <= len(option) <= 300 for option in options):
        raise ExamPaperInvalid(f"{where}: every option must be 1-300 characters.")
    if len({_same_key(option) for option in options}) != len(options):
        raise ExamPaperInvalid(f"{where}: options must all be different.")
    if any(quizzes._CATCH_ALL.match(option) for option in options):
        raise ExamPaperInvalid(f'{where}: options such as "All of the above" are not allowed.')
    if question.answer_index is None or not 0 <= question.answer_index < len(options):
        raise ExamPaperInvalid(f"{where}: answer_index must point at the correct option.")
    return options


def _question(draft: ExamDraftQuestion, plan: BlueprintSection | None, question_type: str,
              sources: dict, where: str) -> ExamQuestion:
    """One checked question. ``plan`` is the blueprint section, or None for a custom structure."""
    text = _clean(draft.text)
    if not text:
        raise ExamPaperInvalid(f"{where}: question text is empty.")
    options = _options(draft, question_type, plan.options_per_question if plan else None, where)
    if question_type not in ("mcq", "true_false") and not _clean(draft.answer):
        raise ExamPaperInvalid(f"{where}: write the answer or marking points in 'answer'.")
    subparts = [ExamSubpart(text=_clean(part.text), marks=part.marks if part.marks and part.marks > 0
                            else None) for part in draft.subparts if _clean(part.text)]
    if question_type in ("mcq", "true_false"):
        subparts = []
    if len({_same_key(part.text) for part in subparts}) != len(subparts):
        raise ExamPaperInvalid(f"{where}: sub-parts must all be different.")
    expected = plan.subparts_per_question if plan else None
    if expected is not None:
        if expected == 0 and subparts:
            raise ExamPaperInvalid(f"{where}: this section's questions have no sub-parts.")
        if expected and not max(2, expected - 1) <= len(subparts) <= expected + 1:
            raise ExamPaperInvalid(f"{where}: needs {expected} sub-parts.")

    marks = draft.marks if draft.marks and draft.marks > 0 else None
    target = plan.marks_per_question if plan else None
    if subparts and all(part.marks for part in subparts):
        marks = round(sum(part.marks for part in subparts), 2)
        if target is not None and not _close(marks, target):
            raise ExamPaperInvalid(f"{where}: sub-part marks must add up to {target:g}.")
    elif subparts and target is not None:
        raise ExamPaperInvalid(f"{where}: give every sub-part its marks (adding up to {target:g}).")
    if target is not None:
        marks = target
    return ExamQuestion(text=text, marks=marks, options=options, subparts=subparts,
                        source=quizzes._source_for(draft.source, sources))


def _total(sections: list[ExamSection]) -> float | None:
    """Marks a candidate can score: answered questions only; None if any mark is unknown."""
    total = 0.0
    for section in sections:
        marks = [question.marks for question in section.questions]
        if any(mark is None for mark in marks):
            return None
        counted = sorted(marks, reverse=True)[:section.answer_count] if section.answer_count else marks
        total += sum(counted)
    return round(total, 2)


def _section_title(index: int) -> str:
    return f"Section {chr(ord('A') + index)}" if index < 26 else f"Section {index + 1}"


def build_paper(draft: ExamDraft, plan: ExamBlueprint, matches: list[dict],
                fallback_title: str) -> ExamPaper:
    """Check the draft against the merged blueprint and build the printable paper.

    Raises ``ExamPaperInvalid`` or ``ValidationError``. Following the plan, the
    blueprint fixes the sections, their order, question types, counts, marks
    and instructions; the model only supplies question text. A custom
    structure (the student asked for one) keeps the model's sections but still
    checks every question. Duration and instructions are never invented.
    """
    sources = quizzes._passage_sources(matches)
    drafts = [section for section in draft.sections if section.questions]
    if not drafts:
        raise ExamPaperInvalid("The paper has no questions.")
    custom = draft.custom_structure
    sections = []
    if not custom:
        if len(drafts) != len(plan.sections):
            raise ExamPaperInvalid(f"The paper needs exactly {len(plan.sections)} sections.")
        for index, (planned, written) in enumerate(zip(plan.sections, drafts)):
            where = planned.title or _section_title(index)
            if len(written.questions) != planned.question_count:
                raise ExamPaperInvalid(f"{where}: needs exactly {planned.question_count} questions.")
            questions = [_question(question, planned, planned.question_type, sources,
                                   f"{where} question {number}")
                         for number, question in enumerate(written.questions, 1)]
            sections.append(ExamSection(
                title=planned.title or _section_title(index), instructions=planned.instructions,
                question_type=planned.question_type, answer_count=planned.answer_count,
                subpart_style=planned.subpart_style if planned.subparts_per_question else "none",
                questions=questions))
    else:
        if sum(len(section.questions) for section in drafts) > MAX_QUESTIONS:
            raise ExamPaperInvalid(f"The paper can have at most {MAX_QUESTIONS} questions.")
        for index, written in enumerate(drafts[:MAX_SECTIONS]):
            question_type = written.question_type if written.question_type in QuestionType.__args__ \
                else "short_answer"
            where = _clean(written.title) or _section_title(index)
            questions = [_question(question, None, question_type, sources,
                                   f"{where} question {number}")
                         for number, question in enumerate(written.questions, 1)]
            sections.append(ExamSection(
                title=where[:120], instructions=_clean(written.instructions)[:400],
                question_type=question_type,
                subpart_style="a_b_c" if any(q.subparts for q in questions) else "none",
                questions=questions))
    texts = [_same_key(question.text) for section in sections for question in section.questions]
    if len(set(texts)) != len(texts):
        raise ExamPaperInvalid("The paper contains duplicate questions.")
    return ExamPaper(
        title=_clean(draft.title)[:160] or _clean(fallback_title)[:160] or "Practice exam",
        duration_minutes=None if custom else plan.duration_minutes,
        total_marks=_total(sections),
        general_instructions=[] if custom else plan.general_instructions,
        sections=sections,
        follows_past_papers=not custom,
    )


def cited_pages(paper: ExamPaper) -> set[tuple[str, int | None]]:
    """(filename, page) of every question's source."""
    return {(question.source.filename, question.source.page)
            for section in paper.sections for question in section.questions if question.source}


def describe_problem(exc: Exception) -> str:
    """Short, content-free reason a draft was rejected (for the retry prompt and logs)."""
    if isinstance(exc, ValidationError):
        return quizzes.describe_problem(exc)
    return str(exc)[:600]


# --------------------------------------------------------------------------- sufficiency
def enough_material(matches: list[dict], settings: Settings | None = None) -> bool:
    """Deterministic minimum before any LLM call: enough relevant chunks and text."""
    settings = settings or get_settings()
    chunks, chars = quizzes._material(matches)
    return (chunks >= max(1, settings.exam_min_chunks)
            and chars >= max(0, settings.exam_min_context_chars))


# --------------------------------------------------------------------------- generation
def build_messages(question: str, context: str, history: str, plan: ExamBlueprint,
                   problem: str = "") -> list[dict[str, str]]:
    """Messages for the structured exam call: grounding rules plus the task prompt."""
    grounding = CORE_DIR / "grounding.md"
    rules = _read_prompt(str(grounding)) if grounding.is_file() else ""
    system = ("You write original university exam questions in a given paper format. "
              "Return only valid JSON.\n\n" + rules).strip()
    prompt = _fill(_read_prompt(str(PROMPT_FILE)), blueprint=describe(plan), context=context,
                   history=history, question=question)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if problem:
        messages.append({"role": "user", "content": (
            f"Your previous paper could not be used: {problem}\n"
            "Write the whole paper again and follow the required structure exactly: the same "
            "sections in the same order, the exact number of questions in each, the stated "
            "sub-parts and marks, and the stated number of options for multiple-choice questions.")})
    return messages


async def generate_draft(question: str, context: str, history: list[dict], plan: ExamBlueprint,
                         provider: str, model: str, fallback: tuple[str, str] | None = None,
                         problem: str = "") -> ExamDraft:
    """Ask the configured model for the paper. Raises ProviderError on failure."""
    messages = build_messages(question, context, llm_provider._history_text(history), plan, problem)
    return await llm_provider.generate_structured(messages, ExamDraft, provider, model,
                                                  fallback=fallback)


# --------------------------------------------------------------------------- student messages
def _period(ttl_minutes: int) -> str:
    if ttl_minutes % 60 == 0 and ttl_minutes >= 60:
        hours = ttl_minutes // 60
        return f"{hours} hour" + ("s" if hours != 1 else "")
    return f"{ttl_minutes} minute" + ("s" if ttl_minutes != 1 else "")


def ready_message(ttl_minutes: int, paper: ExamPaper) -> str:
    """Success message shown with the Download PDF button. Never contains questions."""
    basis = ("It follows the format of your lecturer's past papers, with new questions "
             "written from your course materials." if paper.follows_past_papers
             else "It uses the structure you asked for, with new questions written from your "
                  "course materials.")
    return (
        "Your practice exam paper is ready!\n\n"
        f"{basis} It has no answer key.\n\n"
        "Click the button below to download your PDF. Please save it to your device: generated "
        f"papers are only available for {_period(ttl_minutes)} and are then deleted automatically."
    )


def limited_message(question: str) -> str:
    """Honest reply when the course material is too thin for a full paper."""
    topic = quizzes._topic(question)
    return (f"There isn't enough course material about \"{topic}\" to write a full practice "
            "paper in your lecturer's exam format, so I haven't created one. Try /exam with a "
            "wider topic, such as a whole chapter or the full course.")


def failed_message() -> str:
    """Reply when no valid paper came back after the retry."""
    return ("Sorry, I couldn't write a practice paper that matches your lecturer's exam format "
            "this time, so I haven't created one. Please try /exam again.")


def render_failed_message() -> str:
    """Reply when the PDF could not be produced."""
    return "Sorry, I couldn't create the exam paper PDF this time. Please try /exam again."
